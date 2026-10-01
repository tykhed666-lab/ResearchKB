"""验证、保存并入库用户上传的PDF。"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pymupdf

from research_kb.chunker import split_pages_into_chunks
from research_kb.document_registry import (
    STATUS_INDEXED,
    DocumentMetadata,
    DocumentRegistry,
)
from research_kb.indexer import (
    EvidenceIndexer,
    calculate_document_hash,
)
from research_kb.pdf_loader import load_pdf_pages
from research_kb.settings import RAW_DATA_DIR
from research_kb.text_cleaner import clean_pdf_pages

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_PAGES = 30


@dataclass(frozen=True, slots=True)
class UploadReport:
    """记录一次 PDF 上传和入库的结果。"""

    # SQLite documents 表中的业务 ID。
    document_id: str

    # PDF 在 data/raw 中实际保存的文件名。
    saved_name: str

    total_pages: int
    processed_pages: int
    empty_pages: int
    evidence_chunks: int
    inserted_chunks: int
    skipped_chunks: int
    truncated: bool
    renamed: bool

    # True 表示相同内容已经成功入库，
    # 本次没有重新解析和调用 Embedding。
    duplicate: bool


class PdfUploadService:
    """处理Streamlit上传的PDF文件。"""

    def __init__(
        self,
        indexer: EvidenceIndexer,
        registry: DocumentRegistry,
    ) -> None:
        """保存证据入库服务和文档注册表。"""
        self.indexer = indexer
        self.registry = registry

    def ingest_pdf(
        self,
        file_name: str,
        file_bytes: bytes,
        metadata: DocumentMetadata | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        force_reindex: bool = False,
    ) -> UploadReport:
        """验证、登记、保存、解析并入库一个 PDF。

        Args:
            file_name:
                用户上传时的原始文件名。
            file_bytes:
                PDF 文件的二进制内容。
            metadata:
                标题、公司、行业等文档级信息。
                未提供时使用文件名生成默认信息。
            max_pages:
                本次最多处理的物理页数。
            force_reindex:
                是否强制重新执行解析和入库。
                已存在的证据 ID 仍会被 indexer 跳过。

        Returns:
            本次上传与入库的结果。

        Raises:
            ValueError:
                文件类型、大小、页数或内容不正确。
        """
        if max_pages <= 0:
            raise ValueError("max_pages 必须大于 0")

        if not file_bytes:
            raise ValueError("上传文件为空")

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("PDF 不能超过 20MB")

        # Path.name 会移除用户传入的目录部分，
        # 防止文件被写到 data/raw 以外。
        safe_name = Path(file_name).name

        if not safe_name:
            raise ValueError("PDF 文件名不能为空")

        if Path(safe_name).suffix.lower() != ".pdf":
            raise ValueError("只支持 PDF 文件")

        # 先验证 PDF，避免无效内容进入登记簿。
        total_pages = self._validate_pdf(file_bytes)

        # 直接根据上传字节计算哈希，
        # 不需要先把文件写入磁盘。
        uploaded_hash = sha256(file_bytes).hexdigest()

        existing = self.registry.get_by_hash(uploaded_hash)

        # 同一内容已经成功入库时直接返回。
        # 这是文档级去重，比只依靠证据 ID 更早拦截，
        # 因而不会再次解析 PDF 或请求 Embedding。
        if (
            existing is not None
            and existing.status == STATUS_INDEXED
            and not force_reindex
        ):
            return UploadReport(
                document_id=existing.document_id,
                saved_name=existing.saved_name,
                total_pages=existing.total_pages,
                processed_pages=(existing.processed_pages),
                empty_pages=0,
                evidence_chunks=(existing.text_chunk_count),
                inserted_chunks=0,
                skipped_chunks=(existing.text_chunk_count),
                truncated=(existing.processed_pages < existing.total_pages),
                renamed=(existing.saved_name != safe_name),
                duplicate=True,
            )

        if existing is None:
            destination, renamed = self._choose_destination(
                safe_name=safe_name,
                file_bytes=file_bytes,
            )

            # 没有从页面填写元数据时，
            # 使用文件名和“未分类”生成兼容记录。
            effective_metadata = metadata or DocumentMetadata(
                title=Path(safe_name).stem,
                document_type="未分类",
            )

            record, _ = self.registry.register_document(
                document_hash=uploaded_hash,
                original_name=safe_name,
                saved_name=destination.name,
                metadata=effective_metadata,
                total_pages=total_pages,
            )

        else:
            # failed、saved 或 processing 状态允许重试。
            record = existing
            destination = RAW_DATA_DIR / record.saved_name
            renamed = record.saved_name != safe_name

        # 从这里开始，页面可以看到文档正在处理。
        self.registry.mark_processing(record.document_id)

        try:
            # 文件写入也属于入库流程。把它放在 try 中，
            # 磁盘错误同样会在注册表中留下 failed 状态。
            # 已存在的相同文件不会重复写入。
            if not destination.exists():
                destination.write_bytes(file_bytes)

            raw_pages = load_pdf_pages(destination)

            selected_pages = raw_pages[:max_pages]

            cleaned_pages = clean_pdf_pages(selected_pages)

            empty_pages = sum(page.is_empty for page in cleaned_pages)

            evidence_chunks = split_pages_into_chunks(cleaned_pages)

            if not evidence_chunks:
                raise ValueError("选定页面没有可入库的文本，当前版本暂不支持纯扫描 PDF")

            indexing_report = self.indexer.index_chunks(
                evidence_chunks,
                batch_size=10,
            )

            # Milvus 入库成功后，才将 SQLite
            # 中的状态改为 indexed。
            self.registry.mark_indexed(
                document_id=record.document_id,
                total_pages=total_pages,
                processed_pages=len(selected_pages),
                text_chunk_count=len(evidence_chunks),
                # 重新入库文本时保留已有图像统计。
                image_chunk_count=(record.image_chunk_count),
            )

        except Exception as error:
            # 保留文档记录和失败原因，
            # 方便页面展示并允许稍后重试。
            self.registry.mark_failed(
                document_id=record.document_id,
                error_message=(f"{type(error).__name__}: {error}"),
            )
            raise

        return UploadReport(
            document_id=record.document_id,
            saved_name=destination.name,
            total_pages=total_pages,
            processed_pages=len(selected_pages),
            empty_pages=empty_pages,
            evidence_chunks=len(evidence_chunks),
            inserted_chunks=(indexing_report.inserted_chunks),
            skipped_chunks=(indexing_report.skipped_chunks),
            truncated=(total_pages > len(selected_pages)),
            renamed=renamed,
            duplicate=False,
        )

    @staticmethod
    def _validate_pdf(file_bytes: bytes) -> int:
        """使用PyMuPDF检查上传内容是否为可读PDF。"""
        try:
            with pymupdf.open(
                stream=file_bytes,
                filetype="pdf",
            ) as document:
                if document.needs_pass:
                    raise ValueError("暂不支持带密码的PDF")

                if document.page_count <= 0:
                    raise ValueError("PDF中没有页面")

                return document.page_count

        except ValueError:
            raise

        except Exception as error:
            raise ValueError("上传内容不是有效的PDF") from error

    @staticmethod
    def _choose_destination(
        safe_name: str,
        file_bytes: bytes,
    ) -> tuple[Path, bool]:
        """避免同名但内容不同的PDF互相覆盖。"""
        RAW_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = RAW_DATA_DIR / safe_name

        if not destination.exists():
            return destination, False

        uploaded_hash = sha256(file_bytes).hexdigest()
        existing_hash = calculate_document_hash(destination)

        if uploaded_hash == existing_hash:
            return destination, False

        original_path = Path(safe_name)
        renamed_name = f"{original_path.stem}_{uploaded_hash[:8]}.pdf"

        return RAW_DATA_DIR / renamed_name, True
