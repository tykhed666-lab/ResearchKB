"""测试研究项目隔离和文档维护流程。"""

from pathlib import Path

import pytest

from research_kb.document_registry import (
    DocumentMetadata,
    DocumentRegistry,
)
from research_kb.document_service import (
    DocumentManagementService,
)
from research_kb.upload_service import (
    UploadReport,
)


class FakeMilvusStore:
    """记录删除请求，不连接正式 Milvus。"""

    def __init__(self) -> None:
        self.delete_calls: list[tuple[str, str | None]] = []

    def delete_source_records(
        self,
        source: str,
        content_type: str | None = None,
    ) -> int:
        """模拟删除并返回删除前的证据数量。"""
        self.delete_calls.append((source, content_type))

        return 4 if content_type == "text" else 5


class FakeUploadService:
    """记录重新入库参数，不调用 Embedding。"""

    def __init__(self) -> None:
        self.ingest_calls: list[dict] = []

    def ingest_pdf(
        self,
        **parameters,
    ) -> UploadReport:
        """模拟重新入库成功。"""
        self.ingest_calls.append(parameters)

        return UploadReport(
            document_id="test-document",
            saved_name=parameters["file_name"],
            total_pages=10,
            processed_pages=(parameters["max_pages"]),
            empty_pages=0,
            evidence_chunks=4,
            inserted_chunks=4,
            skipped_chunks=0,
            truncated=False,
            renamed=False,
            duplicate=False,
        )


def register_test_document(
    registry: DocumentRegistry,
    *,
    document_hash: str,
    saved_name: str,
    title: str,
    industry: str,
    company: str,
    report_date: str,
):
    """创建一条测试文档，减少测试中的重复代码。"""
    document, created = registry.register_document(
        document_hash=document_hash,
        original_name=saved_name,
        saved_name=saved_name,
        metadata=DocumentMetadata(
            title=title,
            document_type="年报",
            industry=industry,
            company=company,
            report_date=report_date,
        ),
        total_pages=10,
    )

    assert created is True
    return document


def test_project_assignment_and_combined_filters(
    tmp_path: Path,
) -> None:
    """项目、行业、公司和日期能够组合限制文档范围。"""
    registry = DocumentRegistry(tmp_path / "registry.db")

    ai_project = registry.create_project("AI 基础设施")
    retail_project = registry.create_project("消费零售")

    ai_document = register_test_document(
        registry,
        document_hash="a" * 64,
        saved_name="ai.pdf",
        title="AI报告",
        industry="半导体",
        company="Example AI",
        report_date="2026-03-01",
    )
    retail_document = register_test_document(
        registry,
        document_hash="b" * 64,
        saved_name="retail.pdf",
        title="零售报告",
        industry="消费零售",
        company="Example Retail",
        report_date="2025-03-01",
    )

    registry.assign_document(
        ai_document.document_id,
        ai_project.project_id,
    )
    registry.assign_document(
        retail_document.document_id,
        retail_project.project_id,
    )

    filtered = registry.list_documents(
        project_id=ai_project.project_id,
        industry="半导体",
        company="Example AI",
        report_date_from="2026-01-01",
        report_date_to="2026-12-31",
    )

    assert [document.document_id for document in filtered] == [ai_document.document_id]

    with pytest.raises(
        ValueError,
        match="项目名称已经存在",
    ):
        registry.create_project("ai 基础设施")


def test_delete_document_cleans_all_storage(
    tmp_path: Path,
) -> None:
    """完整删除会同步清理向量、登记记录和本地文件。"""
    raw_directory = tmp_path / "raw"
    image_root = tmp_path / "images"
    raw_directory.mkdir()
    image_root.mkdir()

    registry = DocumentRegistry(tmp_path / "registry.db")
    document = register_test_document(
        registry,
        document_hash="c" * 64,
        saved_name="delete.pdf",
        title="删除测试",
        industry="测试行业",
        company="Example",
        report_date="2025-01-01",
    )

    pdf_path = raw_directory / "delete.pdf"
    pdf_path.write_bytes(b"test-pdf")

    image_directory = image_root / "delete"
    image_directory.mkdir()
    (image_directory / "page_0001.png").write_bytes(b"test-image")

    store = FakeMilvusStore()
    service = DocumentManagementService(
        store=store,
        registry=registry,
        upload_service=FakeUploadService(),
        raw_data_dir=raw_directory,
        image_root=image_root,
        backup_root=tmp_path / "backups",
    )

    report = service.delete_document(document.document_id)

    assert store.delete_calls == [("delete.pdf", None)]
    assert report.deleted_evidence_count == 5
    assert (report.backup_directory / "document.json").is_file()
    assert (report.backup_directory / "delete.pdf").is_file()
    assert (report.backup_directory / "images" / "page_0001.png").is_file()
    assert report.pdf_deleted is True
    assert report.image_directory_deleted is True
    assert report.cleanup_warnings == ()
    assert not pdf_path.exists()
    assert not image_directory.exists()
    assert registry.get_by_id(document.document_id) is None


def test_reindex_replaces_only_text_evidence(
    tmp_path: Path,
) -> None:
    """重新入库只删除文本证据并强制执行上传流程。"""
    raw_directory = tmp_path / "raw"
    image_root = tmp_path / "images"
    raw_directory.mkdir()
    image_root.mkdir()

    registry = DocumentRegistry(tmp_path / "registry.db")
    document = register_test_document(
        registry,
        document_hash="d" * 64,
        saved_name="reindex.pdf",
        title="重新入库测试",
        industry="测试行业",
        company="Example",
        report_date="2025-01-01",
    )

    registry.mark_indexed(
        document_id=document.document_id,
        total_pages=10,
        processed_pages=5,
        text_chunk_count=4,
        image_chunk_count=1,
    )

    pdf_bytes = b"reindex-pdf"
    (raw_directory / "reindex.pdf").write_bytes(pdf_bytes)

    store = FakeMilvusStore()
    upload_service = FakeUploadService()
    service = DocumentManagementService(
        store=store,
        registry=registry,
        upload_service=upload_service,
        raw_data_dir=raw_directory,
        image_root=image_root,
        backup_root=tmp_path / "backups",
    )

    report = service.reindex_document(
        document_id=document.document_id,
        max_pages=3,
    )

    assert store.delete_calls == [("reindex.pdf", "text")]
    assert report.deleted_text_count == 4
    assert len(upload_service.ingest_calls) == 1

    ingest_call = upload_service.ingest_calls[0]
    assert ingest_call["force_reindex"] is True
    assert ingest_call["max_pages"] == 3
    assert ingest_call["file_bytes"] == pdf_bytes
