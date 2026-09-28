"""使用已有 NVIDIA 报告检查上传和文档级去重。"""

from pathlib import Path

from research_kb.document_registry import (
    DocumentRegistry,
    STATUS_INDEXED,
)
from research_kb.embedding_service import (
    EmbeddingService,
)
from research_kb.indexer import (
    EvidenceIndexer,
)
from research_kb.milvus_store import (
    MilvusStore,
)
from research_kb.upload_service import (
    PdfUploadService,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "01_NVDA_2026_Annual_Report.pdf"
)


def main() -> None:
    """验证相同 PDF 不会重复登记或生成向量。"""
    store = MilvusStore()

    registry = DocumentRegistry()

    indexer = EvidenceIndexer(
        store=store,
        embedding_service=EmbeddingService(),
    )

    upload_service = PdfUploadService(
        indexer=indexer,
        registry=registry,
    )

    before_entity_count = (
        store.get_entity_count()
    )

    before_document_count = len(
        registry.list_documents()
    )

    report = upload_service.ingest_pdf(
        file_name=PDF_PATH.name,
        file_bytes=PDF_PATH.read_bytes(),
        # 原始 NVIDIA 资料处理了前 20 页。
        max_pages=20,
    )

    after_entity_count = (
        store.get_entity_count()
    )

    after_document_count = len(
        registry.list_documents()
    )

    document = registry.get_by_id(
        report.document_id
    )

    if document is None:
        raise RuntimeError(
            "上传完成后找不到文档登记"
        )

    print(
        f"文档ID：{report.document_id}"
    )
    print(
        f"保存文件名：{report.saved_name}"
    )
    print(
        f"PDF总页数：{report.total_pages}"
    )
    print(
        f"处理页数：{report.processed_pages}"
    )
    print(
        f"证据数量：{report.evidence_chunks}"
    )
    print(
        f"本次新增：{report.inserted_chunks}"
    )
    print(
        f"本次跳过：{report.skipped_chunks}"
    )
    print(
        f"是否重复文档：{report.duplicate}"
    )
    print(
        f"登记状态：{document.status}"
    )
    print(
        "入库前文档数："
        f"{before_document_count}"
    )
    print(
        "入库后文档数："
        f"{after_document_count}"
    )
    print(
        "入库前实体数："
        f"{before_entity_count}"
    )
    print(
        "入库后实体数："
        f"{after_entity_count}"
    )

    # 以下断言保证重复上传没有修改数据库数量，
    # 也没有再次向 Milvus 写入证据。
    assert report.duplicate is True
    assert report.inserted_chunks == 0
    assert document.status == STATUS_INDEXED

    assert (
        before_document_count
        == after_document_count
        == 6
    )

    assert (
        before_entity_count
        == after_entity_count
        == 311
    )

    print(
        "真实重复上传验收通过"
    )


if __name__ == "__main__":
    main()