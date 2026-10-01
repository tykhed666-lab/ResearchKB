"""测试文档注册表和上传流程的核心行为。"""

from pathlib import Path

import pymupdf
import pytest

import research_kb.upload_service as upload_module
from research_kb.chunker import EvidenceChunk
from research_kb.document_registry import (
    STATUS_FAILED,
    STATUS_INDEXED,
    DocumentMetadata,
    DocumentRegistry,
)
from research_kb.indexer import (
    EvidenceIndexer,
    IndexingReport,
)
from research_kb.upload_service import (
    PdfUploadService,
)


class FakeIndexer(EvidenceIndexer):
    """不连接 Milvus 的测试入库器。"""

    def __init__(self) -> None:
        # 测试替身不需要真正的 Milvus 和
        # EmbeddingService，因此不调用父类构造函数。
        self.call_count = 0

    def index_chunks(
        self,
        chunks: list[EvidenceChunk],
        batch_size: int = 10,
    ) -> IndexingReport:
        """模拟所有证据都成功入库。"""
        self.call_count += 1

        return IndexingReport(
            total_chunks=len(chunks),
            inserted_chunks=len(chunks),
            skipped_chunks=0,
        )


class FailingIndexer(EvidenceIndexer):
    """模拟 Milvus 入库失败。"""

    def __init__(self) -> None:
        pass

    def index_chunks(
        self,
        chunks: list[EvidenceChunk],
        batch_size: int = 10,
    ) -> IndexingReport:
        """抛出固定异常，检查失败状态。"""
        raise RuntimeError("模拟向量数据库连接失败")


def create_test_pdf() -> bytes:
    """在内存中创建一页可提取文字的 PDF。"""
    document = pymupdf.open()

    try:
        page = document.new_page()

        page.insert_text(
            (72, 72),
            ("Example company annual report. Revenue increased in 2025."),
        )

        return document.tobytes()

    finally:
        document.close()


def test_registry_deduplicates_and_updates_status(
    tmp_path: Path,
) -> None:
    """相同哈希只登记一次，并能更新处理状态。"""
    registry = DocumentRegistry(tmp_path / "registry.db")

    metadata = DocumentMetadata(
        title="测试年报",
        document_type="年报",
        company="Example",
        ticker="exm",
        industry="零售",
        report_date="2025-12-31",
    )

    first, first_created = registry.register_document(
        document_hash="a" * 64,
        original_name="report.pdf",
        saved_name="report.pdf",
        metadata=metadata,
        total_pages=100,
    )

    second, second_created = registry.register_document(
        document_hash="a" * 64,
        original_name="copy.pdf",
        saved_name="copy.pdf",
        metadata=metadata,
        total_pages=100,
    )

    assert first_created is True
    assert second_created is False
    assert first.document_id == second.document_id
    assert len(registry.list_documents()) == 1

    registry.mark_processing(first.document_id)

    registry.mark_indexed(
        document_id=first.document_id,
        total_pages=100,
        processed_pages=30,
        text_chunk_count=88,
        image_chunk_count=2,
    )

    indexed = registry.get_by_id(first.document_id)

    assert indexed is not None
    assert indexed.status == STATUS_INDEXED
    assert indexed.ticker == "EXM"
    assert indexed.text_chunk_count == 88
    assert indexed.image_chunk_count == 2


def test_upload_skips_duplicate_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相同 PDF 第二次上传时不再调用入库器。"""
    raw_directory = tmp_path / "raw"

    # upload_service 中的 RAW_DATA_DIR 是模块变量。
    # monkeypatch 只在本测试期间把它换成临时目录。
    monkeypatch.setattr(
        upload_module,
        "RAW_DATA_DIR",
        raw_directory,
    )

    registry = DocumentRegistry(tmp_path / "registry.db")

    fake_indexer = FakeIndexer()

    upload_service = PdfUploadService(
        indexer=fake_indexer,
        registry=registry,
    )

    pdf_bytes = create_test_pdf()

    metadata = DocumentMetadata(
        title="Example 2025 Annual Report",
        document_type="年报",
        company="Example",
        ticker="EXM",
        industry="零售",
        report_date="2025-12-31",
    )

    first_report = upload_service.ingest_pdf(
        file_name="example.pdf",
        file_bytes=pdf_bytes,
        metadata=metadata,
        max_pages=1,
    )

    second_report = upload_service.ingest_pdf(
        file_name="renamed_copy.pdf",
        file_bytes=pdf_bytes,
        metadata=metadata,
        max_pages=1,
    )

    assert first_report.duplicate is False
    assert first_report.inserted_chunks > 0

    assert second_report.duplicate is True
    assert second_report.inserted_chunks == 0

    # 入库器只在第一次上传时调用一次。
    assert fake_indexer.call_count == 1

    # 相同内容即使文件名不同，也只有一条文档记录。
    assert len(registry.list_documents()) == 1

    assert first_report.document_id == second_report.document_id


def test_upload_failure_is_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """向量入库失败后，SQLite 保留失败原因。"""
    monkeypatch.setattr(
        upload_module,
        "RAW_DATA_DIR",
        tmp_path / "raw",
    )

    registry = DocumentRegistry(tmp_path / "registry.db")

    upload_service = PdfUploadService(
        indexer=FailingIndexer(),
        registry=registry,
    )

    with pytest.raises(
        RuntimeError,
        match="模拟向量数据库连接失败",
    ):
        upload_service.ingest_pdf(
            file_name="failed.pdf",
            file_bytes=create_test_pdf(),
            metadata=DocumentMetadata(
                title="失败测试",
                document_type="年报",
            ),
            max_pages=1,
        )

    documents = registry.list_documents()

    assert len(documents) == 1
    assert documents[0].status == STATUS_FAILED
    assert documents[0].error_message is not None
    assert "模拟向量数据库连接失败" in documents[0].error_message
