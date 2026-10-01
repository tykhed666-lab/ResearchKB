"""测试 SEC HTML 可见正文提取和证据切分。"""

from research_kb.external_source_registry import ExternalSourceRecord
from research_kb.sec_html import (
    extract_visible_sec_text,
    split_sec_text_into_chunks,
)


def build_record() -> ExternalSourceRecord:
    """构造无需数据库的外部来源记录。"""
    return ExternalSourceRecord(
        source_id="sec_0000027419_demo",
        provider="sec_edgar",
        ticker="TGT",
        cik="0000027419",
        company_name="TARGET CORP",
        form_type="10-K",
        report_date="2026-01-31",
        filing_date="2026-03-11",
        accession_number="0000027419-26-000016",
        primary_document="annual.htm",
        document_url="https://www.sec.gov/Archives/annual.htm",
        index_url="https://www.sec.gov/Archives/index.html",
        collected_at="2026-09-30T00:00:00+00:00",
        status="saved",
        project_id=None,
        local_path=None,
        evidence_count=0,
        indexed_at=None,
        error_message=None,
    )


def test_extract_visible_text_skips_hidden_xbrl() -> None:
    """脚本、样式和隐藏 XBRL 不应进入正文。"""
    html = b"""
    <html><head><style>secret-style</style></head><body>
      <h1>Item 1. Business</h1>
      <p>Visible operating priorities.</p>
      <ix:hidden><span>duplicate machine fact</span></ix:hidden>
      <div style="display:none">hidden note</div>
      <script>secret-script</script>
    </body></html>
    """

    text = extract_visible_sec_text(html)

    assert "Item 1. Business" in text
    assert "Visible operating priorities." in text
    assert "duplicate machine fact" not in text
    assert "hidden note" not in text
    assert "secret-script" not in text


def test_split_sec_text_uses_external_semantics() -> None:
    """SEC 证据页码为 0，并携带报告期上下文。"""
    chunks, truncated = split_sec_text_into_chunks(
        record=build_record(),
        text=("Revenue increased. " * 30),
        chunk_size=120,
        chunk_overlap=20,
        max_chunks=2,
    )

    assert truncated is True
    assert len(chunks) == 2
    assert all(chunk.page_number == 0 for chunk in chunks)
    assert all(chunk.content_type == "sec_html" for chunk in chunks)
    assert all(chunk.source == "sec_0000027419_demo" for chunk in chunks)
    assert "报告期 2026-01-31" in chunks[0].text
