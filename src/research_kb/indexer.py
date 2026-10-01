"""把证据文本转换成向量并写入Milvus。"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from research_kb.chunker import EvidenceChunk
from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.settings import RAW_DATA_DIR


@dataclass(frozen=True, slots=True)
class IndexingReport:
    """记录一次Milvus入库的执行结果."""

    total_chunks: int
    inserted_chunks: int
    skipped_chunks: int


def calculate_document_hash(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """分块计算文件的SHA-256。

    分块读取可以避免把整个PDF一次性加载到内存。

    Args:
        file_path: 原始PDF路径。
        block_size: 每次读取的字节数量。

    Returns:
        64位SHA-256十六进制字符串。
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到原始PDF：{file_path}")

    file_hasher = sha256()
    # 以二进制只读打开文件
    # 每次读一小块，读到空为止
    # 把这一块数据喂给哈希器，累加计算
    with file_path.open("rb") as file:
        while block := file.read(block_size):
            file_hasher.update(block)
    # 返回十六进制字符串
    return file_hasher.hexdigest()


class EvidenceIndexer:
    """负责证据去重、向量化和Milvus入库。"""

    def __init__(
        self,
        store: MilvusStore,
        embedding_service: EmbeddingService,
    ) -> None:
        """保存Milvus和Embedding服务。"""
        self.store = store
        self.embedding_service = embedding_service

    def index_chunks(
        self,
        chunks: list[EvidenceChunk],
        batch_size: int = 10,
        source_file_paths: (Mapping[str, str | Path] | None) = None,
    ) -> IndexingReport:
        """批量把证据写入Milvus。

        已经存在的evidence_id会被跳过，避免重复请求
        Embedding API和重复保存实体。

        Args:
            chunks: 等待入库的证据。
            batch_size: 每批处理的证据数量。
            source_file_paths:
                可选的来源文件路径映射。PDF 不传时仍从
                data/raw 读取；SEC HTML 等外部资料通过该映射
                提供真实文件路径，用于计算稳定文件哈希。

        Returns:
            本次入库的统计报告。
        """
        if batch_size <= 0:
            raise ValueError("batch_size必须大于0")

        if not chunks:
            return IndexingReport(
                total_chunks=0,
                inserted_chunks=0,
                skipped_chunks=0,
            )

        self.store.ensure_collection()
        self.store.load_collection()

        explicit_paths = {
            source: Path(path) for source, path in (source_file_paths or {}).items()
        }

        # 每个来源只计算一次文件哈希。旧 PDF 调用保持兼容，
        # 外部 HTML 则使用调用者明确传入的本地文件。
        source_hashes = {
            source: calculate_document_hash(
                explicit_paths.get(
                    source,
                    RAW_DATA_DIR / source,
                )
            )
            for source in {
                # 提取不重复的PDF名
                chunk.source
                for chunk in chunks
            }
        }

        inserted_count = 0
        skipped_count = 0

        for start_index in range(0, len(chunks), batch_size):
            batch = chunks[start_index : start_index + batch_size]

            existing_ids = self.store.get_existing_ids(
                [chunk.evidence_id for chunk in batch]
            )

            pending_chunks = [
                chunk for chunk in batch if chunk.evidence_id not in existing_ids
            ]

            skipped_count += len(batch) - len(pending_chunks)

            if pending_chunks:
                vectors = self.embedding_service.embed_documents(
                    [chunk.text for chunk in pending_chunks],
                    batch_size=batch_size,
                )

                records = [
                    {
                        "evidence_id": chunk.evidence_id,
                        "vector": vector,
                        "source": chunk.source,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "content_type": chunk.content_type,
                        "document_hash": source_hashes[chunk.source],
                        "text": chunk.text,
                    }
                    for chunk, vector in zip(
                        pending_chunks,
                        vectors,
                        strict=True,
                    )
                ]

                self.store.upsert_records(records)
                inserted_count += len(records)

            processed_count = min(
                start_index + batch_size,
                len(chunks),
            )

            print(
                f"入库进度：{processed_count}/{len(chunks)}，"
                f"新增{inserted_count}，跳过{skipped_count}"
            )

        self.store.flush()

        return IndexingReport(
            total_chunks=len(chunks),
            inserted_chunks=inserted_count,
            skipped_chunks=skipped_count,
        )
