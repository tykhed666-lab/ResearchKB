"""把视觉描述生成的图像证据写入Milvus。"""

from research_kb.embedding_service import EmbeddingService
from research_kb.indexer import EvidenceIndexer
from research_kb.milvus_store import MilvusStore
from research_kb.visual_evidence import (
    load_visual_evidence_chunks,
)


def main() -> None:
    """加载、展示并入库图像证据。"""
    chunks = load_visual_evidence_chunks()

    print(f"待处理图像证据：{len(chunks)}")

    for chunk in chunks:
        print(
            f"{chunk.evidence_id} | "
            f"{chunk.source} | "
            f"PDF第{chunk.page_number}页 | "
            f"{chunk.content_type}"
        )

    store = MilvusStore()
    before_count = store.get_entity_count()

    indexer = EvidenceIndexer(
        store=store,
        embedding_service=EmbeddingService(),
    )

    report = indexer.index_chunks(
        chunks=chunks,
        batch_size=3,
    )

    after_count = store.get_entity_count()

    image_records = store.client.query(
        collection_name=store.collection_name,
        filter='content_type == "image"',
        limit=100,
        output_fields=[
            "evidence_id",
            "source",
            "page_number",
            "content_type",
        ],
    )

    print("\n图像证据入库完成：")
    print(f"处理数量：{report.total_chunks}")
    print(f"本次新增：{report.inserted_chunks}")
    print(f"本次跳过：{report.skipped_chunks}")
    print(f"入库前实体：{before_count}")
    print(f"入库后实体：{after_count}")
    print(f"Milvus图像证据：{len(image_records)}")

    for record in image_records:
        print(
            f"- {record['evidence_id']} | "
            f"{record['source']} | "
            f"PDF第{record['page_number']}页"
        )


if __name__ == "__main__":
    main()