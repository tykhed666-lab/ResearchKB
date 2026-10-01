"""验证集中配置读取和统一证据上下文格式。"""

import pytest

from research_kb.evidence_context import build_evidence_context
from research_kb.retrieval import RetrievalResult
from research_kb.settings import optional_environment, require_environment


def test_required_environment_reports_all_missing_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少配置时一次列出全部变量，减少逐个修复的往返。"""
    first = "RESEARCH_KB_TEST_REQUIRED_FIRST"
    second = "RESEARCH_KB_TEST_REQUIRED_SECOND"
    monkeypatch.delenv(first, raising=False)
    monkeypatch.delenv(second, raising=False)

    with pytest.raises(ValueError, match=f"{first}, {second}"):
        require_environment(first, second)


def test_environment_helpers_strip_values_and_use_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = "RESEARCH_KB_TEST_REQUIRED_VALUE"
    optional = "RESEARCH_KB_TEST_OPTIONAL_VALUE"
    monkeypatch.setenv(required, "  configured  ")
    monkeypatch.delenv(optional, raising=False)

    assert require_environment(required) == ("configured",)
    assert optional_environment(optional, "fallback") == "fallback"


def test_evidence_context_distinguishes_pdf_and_sec() -> None:
    """模型上下文必须给 PDF 页码和 SEC 链接不同的位置语义。"""
    pdf = RetrievalResult(
        evidence_id="text_pdf",
        score=0.91,
        source="report.pdf",
        page_number=3,
        chunk_index=1,
        content_type="text",
        text="PDF evidence",
        display_title="Annual Report",
    )
    sec = RetrievalResult(
        evidence_id="text_sec",
        score=0.82,
        source="sec_source",
        page_number=0,
        chunk_index=1,
        content_type="sec_html",
        text="SEC evidence",
        source_type="sec",
        display_title="10-K",
        source_url="https://www.sec.gov/example",
    )

    context = build_evidence_context(
        [pdf, sec],
        include_score=True,
    )

    assert "Annual Report" in context
    assert "PDF物理页码：3" in context
    assert "SEC官方原文：https://www.sec.gov/example" in context
    assert "相似度：0.9100" in context
