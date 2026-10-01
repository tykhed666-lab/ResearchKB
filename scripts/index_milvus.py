"""读取证据JSON并写入Milvus。"""

import json
from pathlib import Path

from research_kb.chunker import EvidenceChunk
from research_kb.embedding_service import EmbeddingService
from research_kb.indexer import EvidenceIndexer
from research_kb.milvus_store import MilvusStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = PROJECT_ROOT / "data" / "processed" / "first_pass_evidence.json"


def load_evidence_chunks() -> list[EvidenceChunk]:
    """从JSON读取并恢复EvidenceChunk对象。"""
    if not EVIDENCE_PATH.is_file():
        raise FileNotFoundError(f"找不到证据JSON：{EVIDENCE_PATH}")

    # 把json转换为字典列表
    evidence_data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    # 把字典列表转换为EvidenceChunk对象列表
    return [EvidenceChunk(**item) for item in evidence_data]


def main() -> None:
    """执行Milvus批量入库。"""
    chunks = load_evidence_chunks()

    store = MilvusStore()
    embedding_service = EmbeddingService()
    indexer = EvidenceIndexer(
        store=store,
        embedding_service=embedding_service,
    )

    before_count = store.get_entity_count()

    report = indexer.index_chunks(
        chunks,
        batch_size=10,
    )

    after_count = store.get_entity_count()

    print("\n入库完成：")
    print(f"JSON证据数：{report.total_chunks}")
    print(f"本次新增：{report.inserted_chunks}")
    print(f"本次跳过：{report.skipped_chunks}")
    print(f"入库前实体数：{before_count}")
    print(f"入库后实体数：{after_count}")


if __name__ == "__main__":
    main()
