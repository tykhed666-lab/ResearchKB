"""统一协调文档删除与重新入库。"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from research_kb.document_registry import (
    DocumentMetadata,
    DocumentRecord,
    DocumentRegistry,
)
from research_kb.milvus_store import (
    MilvusStore,
)
from research_kb.page_renderer import (
    IMAGE_ROOT,
)
from research_kb.settings import DATA_DIR
from research_kb.upload_service import (
    DEFAULT_MAX_PAGES,
    RAW_DATA_DIR,
    PdfUploadService,
    UploadReport,
)

DEFAULT_DOCUMENT_BACKUP_ROOT = DATA_DIR / "backups" / "deleted_documents"


@dataclass(frozen=True, slots=True)
class DeleteDocumentReport:
    """记录一次完整文档删除的结果。"""

    document_id: str
    saved_name: str

    # 从Milvus中删除的文本和图像证据总数。
    deleted_evidence_count: int

    # 删除前自动创建的可恢复备份目录。
    backup_directory: Path

    # 文件原本不存在时，这两个字段为False，
    # 但不代表删除操作失败。
    pdf_deleted: bool
    image_directory_deleted: bool

    # 本地文件清理失败不会恢复SQLite和Milvus，
    # 因此把具体原因返回给页面展示。
    cleanup_warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReindexDocumentReport:
    """记录一次文本证据重新入库的结果。"""

    document_id: str
    saved_name: str
    deleted_text_count: int

    # PdfUploadService原有的处理结果，
    # 包含处理页数、新增片段数等信息。
    upload_report: UploadReport


class DocumentManagementService:
    """协调SQLite、Milvus和本地文件。"""

    def __init__(
        self,
        store: MilvusStore,
        registry: DocumentRegistry,
        upload_service: PdfUploadService,
        raw_data_dir: str | Path = RAW_DATA_DIR,
        image_root: str | Path = IMAGE_ROOT,
        backup_root: str | Path = (DEFAULT_DOCUMENT_BACKUP_ROOT),
    ) -> None:
        self.store = store
        self.registry = registry
        self.upload_service = upload_service

        # Path统一处理Windows和Linux路径。
        self.raw_data_dir = Path(raw_data_dir)
        self.image_root = Path(image_root)
        self.backup_root = Path(backup_root)

    def _create_deletion_backup(
        self,
        record: DocumentRecord,
        pdf_path: Path,
        image_directory: Path,
    ) -> Path:
        """备份文档元数据、PDF和已有页面图片。

        备份失败时抛出异常，使后续删除不会开始。
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_directory = self.backup_root / (f"{timestamp}_{record.document_id}")

        try:
            backup_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            # 即使本地PDF或图片已经缺失，也保留SQLite
            # 元数据，方便定位和人工恢复。
            metadata_path = backup_directory / "document.json"
            metadata_path.write_text(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if pdf_path.is_file():
                shutil.copy2(
                    pdf_path,
                    backup_directory / pdf_path.name,
                )

            if image_directory.is_dir():
                shutil.copytree(
                    image_directory,
                    backup_directory / "images",
                )

        except Exception as error:
            # 不留下半成品备份，也不继续执行删除。
            if backup_directory.exists():
                shutil.rmtree(
                    backup_directory,
                    ignore_errors=True,
                )

            raise RuntimeError(f"删除前备份失败，已取消删除：{error}") from error

        return backup_directory

    def delete_document(
        self,
        document_id: str,
    ) -> DeleteDocumentReport:
        """完整删除一份文档。

        执行顺序：
        1. 查询SQLite文档记录；
        2. 删除Milvus证据；
        3. 删除SQLite记录；
        4. 清理本地PDF和页面图片。

        SQLite删除失败时会保留本地PDF，
        以后仍然可以重新执行入库。
        """
        record = self.registry.get_by_id(document_id)

        if record is None:
            raise KeyError(f"找不到文档：{document_id}")

        # Path.name移除可能存在的目录部分，
        # 防止路径离开data/raw。
        safe_saved_name = Path(record.saved_name).name

        pdf_path = self.raw_data_dir / safe_saved_name

        image_directory = self.image_root / Path(safe_saved_name).stem

        # 先创建可恢复备份。任何备份错误都会中止删除，
        # 因而不会出现“数据已删但备份不完整”。
        backup_directory = self._create_deletion_backup(
            record=record,
            pdf_path=pdf_path,
            image_directory=(image_directory),
        )

        # Milvus中的source保存的是saved_name，
        # 因此必须用保存后的文件名删除证据。
        deleted_evidence_count = self.store.delete_source_records(record.saved_name)

        # 先完成核心业务数据删除，再清理原始文件。
        # 如果这里失败，本地PDF和备份仍然保留。
        self.registry.delete_document(record.document_id)

        pdf_deleted = False
        image_directory_deleted = False
        cleanup_warnings: list[str] = []

        try:
            if pdf_path.is_file():
                pdf_path.unlink()
                pdf_deleted = True

        except OSError as error:
            cleanup_warnings.append(f"PDF文件清理失败：{error}")

        try:
            if image_directory.is_dir():
                shutil.rmtree(image_directory)
                image_directory_deleted = True

        except OSError as error:
            cleanup_warnings.append(f"页面图片清理失败：{error}")

        return DeleteDocumentReport(
            document_id=record.document_id,
            saved_name=record.saved_name,
            deleted_evidence_count=(deleted_evidence_count),
            backup_directory=(backup_directory),
            pdf_deleted=pdf_deleted,
            image_directory_deleted=(image_directory_deleted),
            cleanup_warnings=tuple(cleanup_warnings),
        )

    def reindex_document(
        self,
        document_id: str,
        max_pages: int | None = None,
    ) -> ReindexDocumentReport:
        """重新生成一份文档的文本证据。

        图像证据不会被删除，因为生成视觉描述
        需要额外调用视觉模型。
        """
        record = self.registry.get_by_id(document_id)

        if record is None:
            raise KeyError(f"找不到文档：{document_id}")

        safe_saved_name = Path(record.saved_name).name

        pdf_path = self.raw_data_dir / safe_saved_name

        if not pdf_path.is_file():
            raise FileNotFoundError(f"找不到文档对应的本地PDF：{pdf_path}")

        effective_max_pages = (
            max_pages
            if max_pages is not None
            else (record.processed_pages or DEFAULT_MAX_PAGES)
        )

        if effective_max_pages <= 0:
            raise ValueError("重新入库页数必须大于0")

        # 只清理文本证据，保留已经生成的
        # 图表视觉描述及对应图片。
        deleted_text_count = self.store.delete_source_records(
            source=record.saved_name,
            content_type="text",
        )

        metadata = DocumentMetadata(
            title=record.title,
            company=record.company,
            ticker=record.ticker,
            industry=record.industry,
            document_type=(record.document_type),
            report_date=record.report_date,
        )

        # force_reindex跳过“文档已经入库”的快速返回，
        # 重新执行PDF解析、切分、Embedding和Milvus写入。
        upload_report = self.upload_service.ingest_pdf(
            file_name=record.saved_name,
            file_bytes=(pdf_path.read_bytes()),
            metadata=metadata,
            max_pages=(effective_max_pages),
            force_reindex=True,
        )

        return ReindexDocumentReport(
            document_id=record.document_id,
            saved_name=record.saved_name,
            deleted_text_count=(deleted_text_count),
            upload_report=upload_report,
        )
