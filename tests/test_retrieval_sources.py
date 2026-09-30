"""测试 PDF 与 SEC 检索来源的差异化行为。"""

from research_kb.retrieval import (
    MilvusRetriever,
    RetrievalResult,
)


class FailingStore:
    """SEC 证据不应调用其 PDF 同页读取方法。"""

    def get_page_records(self, **kwargs):
        raise AssertionError("SEC 证据不能读取 PDF 同页")


def test_sec_result_skips_pdf_page_expansion() -> None:
    """外部网页页码为 0，证据不足时保持原结果。"""
    retriever = MilvusRetriever.__new__(MilvusRetriever)
    retriever.store = FailingStore()
    result = RetrievalResult(
        evidence_id="text_demo",
        score=0.8,
        source="sec_demo",
        page_number=0,
        chunk_index=1,
        content_type="sec_html",
        text="SEC evidence",
        source_type="sec",
        display_title="TGT 10-K (2026-01-31)",
        source_url="https://www.sec.gov/Archives/demo.htm",
        source_date="2026-01-31",
    )

    expanded = retriever.expand_top_result_page(
        query="Target priority",
        results=[result],
    )

    assert expanded == [result]
    assert result.citation == (
        "TGT 10-K (2026-01-31)，SEC 官方原文"
    )
