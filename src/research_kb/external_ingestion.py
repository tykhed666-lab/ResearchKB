"""下载、解析并索引已保存的 SEC 官方披露。"""

from dataclasses import dataclass
from pathlib import Path

from research_kb.external_source_registry import (
    EXTERNAL_STATUS_INDEXED,
    ExternalSourceRecord,
    ExternalSourceRegistry,
)
from research_kb.indexer import EvidenceIndexer
from research_kb.sec_edgar import SecEdgarClient
from research_kb.sec_html import (
    extract_visible_sec_text,
    split_sec_text_into_chunks,
)
from research_kb.settings import DATA_DIR


EXTERNAL_HTML_DIR = DATA_DIR / "external" / "sec"


@dataclass(frozen=True, slots=True)
class ExternalIngestionReport:
    """记录一份 SEC 披露的正文入库结果。"""

    source_id: str
    local_path: Path
    evidence_count: int
    inserted_count: int
    skipped_count: int
    truncated: bool
    already_indexed: bool


class ExternalIngestionService:
    """编排 SEC 下载、本地保存、正文清洗和向量入库。"""

    def __init__(
        self,
        client: SecEdgarClient,
        registry: ExternalSourceRegistry,
        indexer: EvidenceIndexer,
        html_dir: str | Path = EXTERNAL_HTML_DIR,
    ) -> None:
        self.client = client
        self.registry = registry
        self.indexer = indexer
        self.html_dir = Path(html_dir)

    def ingest(
        self,
        source_id: str,
        max_chunks: int = 500,
    ) -> ExternalIngestionReport:
        """把登记簿中的一条 SEC 披露写入 Milvus。

        已经成功入库的来源直接返回，避免重复下载和 Embedding。
        失败状态允许再次调用，从中断位置安全重试。
        """
        record = self.registry.get_by_id(source_id)
        if record is None:
            raise KeyError(f"找不到外部资料：{source_id}")

        if record.status == EXTERNAL_STATUS_INDEXED:
            return ExternalIngestionReport(
                source_id=record.source_id,
                local_path=Path(record.local_path or ""),
                evidence_count=record.evidence_count,
                inserted_count=0,
                skipped_count=record.evidence_count,
                truncated=False,
                already_indexed=True,
            )

        self.html_dir.mkdir(parents=True, exist_ok=True)
        local_path = self.html_dir / f"{record.source_id}.html"

        try:
            html_content = self.client.download_document(
                record.document_url
            )
            local_path.write_bytes(html_content)
            visible_text = extract_visible_sec_text(html_content)
            chunks, truncated = split_sec_text_into_chunks(
                record=record,
                text=visible_text,
                max_chunks=max_chunks,
            )
            indexing = self.indexer.index_chunks(
                chunks=chunks,
                source_file_paths={
                    record.source_id: local_path,
                },
            )
            self.registry.mark_indexed(
                source_id=record.source_id,
                local_path=local_path,
                evidence_count=len(chunks),
            )
        except Exception as error:
            self.registry.mark_failed(
                source_id=record.source_id,
                error_message=str(error),
                local_path=(
                    local_path
                    if local_path.exists()
                    else None
                ),
            )
            raise

        return ExternalIngestionReport(
            source_id=record.source_id,
            local_path=local_path,
            evidence_count=len(chunks),
            inserted_count=indexing.inserted_chunks,
            skipped_count=indexing.skipped_chunks,
            truncated=truncated,
            already_indexed=False,
        )
