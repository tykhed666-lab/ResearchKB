"""使用已有NVIDIA报告检查PDF上传和去重。"""

from pathlib import Path

from research_kb.embedding_service import EmbeddingService
from research_kb.indexer import EvidenceIndexer
from research_kb.milvus_store import MilvusStore
from research_kb.upload_service import PdfUploadService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "01_NVDA_2026_Annual_Report.pdf"
)


def main() -> None:
    """模拟重复上传NVIDIA前20页。"""
    store = MilvusStore()

    indexer = EvidenceIndexer(
        store=store,
        embedding_service=EmbeddingService(),
    )

    upload_service = PdfUploadService(
        indexer=indexer,
    )

    before_count = store.get_entity_count()

    report = upload_service.ingest_pdf(
        file_name=PDF_PATH.name,
        file_bytes=PDF_PATH.read_bytes(),
        # 原始首轮入库只处理了NVIDIA前20页。
        max_pages=20,
    )

    after_count = store.get_entity_count()

    print(f"保存文件名：{report.saved_name}")
    print(f"PDF总页数：{report.total_pages}")
    print(f"处理页数：{report.processed_pages}")
    print(f"空文本页：{report.empty_pages}")
    print(f"证据数量：{report.evidence_chunks}")
    print(f"本次新增：{report.inserted_chunks}")
    print(f"本次跳过：{report.skipped_chunks}")
    print(f"是否截断：{report.truncated}")
    print(f"是否重命名：{report.renamed}")
    print(f"入库前实体：{before_count}")
    print(f"入库后实体：{after_count}")


if __name__ == "__main__":
    main()