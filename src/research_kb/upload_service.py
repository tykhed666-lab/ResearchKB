"""验证、保存并入库用户上传的PDF。"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pymupdf

from research_kb.chunker import split_pages_into_chunks
from research_kb.indexer import (
    EvidenceIndexer,
    calculate_document_hash,
)
from research_kb.pdf_loader import load_pdf_pages
from research_kb.text_cleaner import clean_pdf_pages


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_PAGES = 30


@dataclass(frozen=True, slots=True)
class UploadReport:
    """记录一次PDF上传和入库的结果。"""

    saved_name: str
    total_pages: int
    processed_pages: int
    empty_pages: int
    evidence_chunks: int
    inserted_chunks: int
    skipped_chunks: int
    truncated: bool
    renamed: bool


class PdfUploadService:
    """处理Streamlit上传的PDF文件。"""

    def __init__(
        self,
        indexer: EvidenceIndexer,
    ) -> None:
        """保存证据入库服务。"""
        self.indexer = indexer

    def ingest_pdf(
        self,
        file_name: str,
        file_bytes: bytes,
        max_pages: int = DEFAULT_MAX_PAGES,
    ) -> UploadReport:
        """验证、保存、解析并入库一个PDF。

        Args:
            file_name: 用户上传的原始文件名。
            file_bytes: 上传文件的二进制内容。
            max_pages: 本次最多处理的物理页数。

        Returns:
            上传和入库结果。

        Raises:
            ValueError: 文件类型、大小、页数或PDF内容不正确。
        """
        if max_pages <= 0:
            raise ValueError("max_pages必须大于0")

        if not file_bytes:
            raise ValueError("上传文件为空")

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError("PDF不能超过20MB")

        # Path.name可以移除用户文件名中的目录部分，
        # 防止文件被写到data/raw以外的位置。
        safe_name = Path(file_name).name

        if not safe_name:
            raise ValueError("PDF文件名不能为空")

        if Path(safe_name).suffix.lower() != ".pdf":
            raise ValueError("只支持PDF文件")

        total_pages = self._validate_pdf(file_bytes)

        destination, renamed = self._choose_destination(
            safe_name=safe_name,
            file_bytes=file_bytes,
        )

        # 相同文件已经存在时不重复写磁盘。
        if not destination.exists():
            destination.write_bytes(file_bytes)

        raw_pages = load_pdf_pages(destination)
        selected_pages = raw_pages[:max_pages]
        cleaned_pages = clean_pdf_pages(selected_pages)

        # 统计本次上传的 PDF 中，经过文本清洗后仍然没有任何文字的页面数量。
        empty_pages = sum(
            page.is_empty
            for page in cleaned_pages
        )

        # 上传流程使用 chunker.py 定义的统一默认值。
        # 以后需要针对某份文档调整时，仍可以显式传入参数。
        evidence_chunks = split_pages_into_chunks(cleaned_pages)


        if not evidence_chunks:
            raise ValueError(
                "选定页面没有可入库的文本，"
                "当前版本暂不支持纯扫描PDF"
            )

        indexing_report = self.indexer.index_chunks(
            evidence_chunks,
            batch_size=10,
        )

        return UploadReport(
            saved_name=destination.name,
            total_pages=total_pages,
            processed_pages=len(selected_pages),
            empty_pages=empty_pages,
            evidence_chunks=len(evidence_chunks),
            inserted_chunks=(
                indexing_report.inserted_chunks
            ),
            skipped_chunks=(
                indexing_report.skipped_chunks
            ),
            truncated=total_pages > len(selected_pages),
            renamed=renamed,
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
                    raise ValueError(
                        "暂不支持带密码的PDF"
                    )

                if document.page_count <= 0:
                    raise ValueError(
                        "PDF中没有页面"
                    )

                return document.page_count

        except ValueError:
            raise

        except Exception as error:
            raise ValueError(
                "上传内容不是有效的PDF"
            ) from error

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
        existing_hash = calculate_document_hash(
            destination
        )

        if uploaded_hash == existing_hash:
            return destination, False

        original_path = Path(safe_name)
        renamed_name = (
            f"{original_path.stem}_"
            f"{uploaded_hash[:8]}.pdf"
        )

        return RAW_DATA_DIR / renamed_name, True