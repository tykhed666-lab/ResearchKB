"""测试 SEC 正文入库编排，不调用真实网络和模型。"""

from pathlib import Path

from research_kb.external_ingestion import ExternalIngestionService
from research_kb.external_source_registry import (
    EXTERNAL_STATUS_INDEXED,
    ExternalSourceRegistry,
)
from research_kb.indexer import IndexingReport
from research_kb.sec_edgar import SecCompany, SecFiling


def build_filing() -> SecFiling:
    """构造测试使用的 Target 年报。"""
    accession = "0000027419-26-000016"
    directory = accession.replace("-", "")
    return SecFiling(
        source_id=f"sec_0000027419_{directory}",
        company=SecCompany(
            ticker="TGT",
            cik="0000027419",
            name="TARGET CORP",
        ),
        form_type="10-K",
        accession_number=accession,
        filing_date="2026-03-11",
        report_date="2026-01-31",
        primary_document="tgt-20260131.htm",
        document_url=(
            "https://www.sec.gov/Archives/edgar/data/27419/"
            f"{directory}/tgt-20260131.htm"
        ),
        index_url=(
            "https://www.sec.gov/Archives/edgar/data/27419/"
            f"{directory}/{accession}-index.html"
        ),
    )


class FakeSecClient:
    """返回固定 HTML 的 SEC 客户端。"""

    calls = 0

    def download_document(self, url: str) -> bytes:
        self.calls += 1
        return (
            b"<html><body><h1>Business</h1><p>"
            + b"Target priority and revenue evidence. " * 30
            + b"</p></body></html>"
        )


class FakeIndexer:
    """记录入库参数并返回固定统计。"""

    def __init__(self) -> None:
        self.chunks = []
        self.paths = {}

    def index_chunks(
        self,
        chunks,
        source_file_paths=None,
    ) -> IndexingReport:
        self.chunks = chunks
        self.paths = dict(source_file_paths or {})
        return IndexingReport(
            total_chunks=len(chunks),
            inserted_chunks=len(chunks),
            skipped_chunks=0,
        )


def test_ingest_sec_html_and_skip_second_run(
    tmp_path: Path,
) -> None:
    """首次入库更新状态，第二次不再下载和向量化。"""
    registry = ExternalSourceRegistry(tmp_path / "research.db")
    saved = registry.save_filing(build_filing()).record
    client = FakeSecClient()
    indexer = FakeIndexer()
    service = ExternalIngestionService(
        client=client,
        registry=registry,
        indexer=indexer,
        html_dir=tmp_path / "html",
    )

    first = service.ingest(saved.source_id, max_chunks=10)
    second = service.ingest(saved.source_id, max_chunks=10)
    updated = registry.get_by_id(saved.source_id)

    assert first.inserted_count > 0
    assert first.local_path.is_file()
    assert saved.source_id in indexer.paths
    assert updated is not None
    assert updated.status == EXTERNAL_STATUS_INDEXED
    assert updated.evidence_count == first.evidence_count
    assert updated.indexed_at is not None
    assert second.already_indexed is True
    assert client.calls == 1
