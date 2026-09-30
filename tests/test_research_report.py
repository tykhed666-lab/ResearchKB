"""测试研究简报的引用校验、Markdown 渲染和保存。"""

from pathlib import Path

import pytest

from research_kb.qa import CitationValidationError
from research_kb.research_report import (
    CitedReportItem,
    ResearchReportDraft,
    ResearchReportService,
)
from research_kb.retrieval import RetrievalResult


def build_results() -> list[RetrievalResult]:
    """构造一条 PDF 和一条 SEC 证据。"""
    return [
        RetrievalResult(
            evidence_id="pdf_1",
            score=0.9,
            source="target.pdf",
            page_number=3,
            chunk_index=1,
            content_type="text",
            text="Target operating priority.",
            display_title="Target 年报",
        ),
        RetrievalResult(
            evidence_id="sec_1",
            score=0.8,
            source="sec_target",
            page_number=0,
            chunk_index=1,
            content_type="sec_html",
            text="Target SEC risk evidence.",
            source_type="sec",
            display_title="TGT 10-K (2026-01-31)",
            source_url=(
                "https://www.sec.gov/Archives/target.htm"
            ),
            source_date="2026-01-31",
        ),
    ]


class FakeRetriever:
    """返回固定检索结果并记录调用参数。"""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class FakeStructuredModel:
    """返回预先构造的结构化草稿。"""

    def __init__(self, draft: ResearchReportDraft) -> None:
        self.draft = draft

    def invoke(self, messages):
        return self.draft


def build_draft(
    evidence_id: str = "pdf_1",
) -> ResearchReportDraft:
    """构造包含全部主要章节的简报草稿。"""
    return ResearchReportDraft(
        title="Target经营重点研究简报",
        core_conclusions=[
            CitedReportItem(
                title="经营重点",
                content="公司强调顾客体验。",
                cited_evidence_ids=[evidence_id],
            )
        ],
        key_facts=[
            CitedReportItem(
                title="报告期",
                content="SEC报告期为2026-01-31。",
                cited_evidence_ids=["sec_1"],
            )
        ],
        company_comparison=[],
        catalysts=[],
        risks=[
            CitedReportItem(
                title="执行风险",
                content="战略执行仍存在风险。",
                cited_evidence_ids=["sec_1"],
            )
        ],
        missing_information=["缺少竞争对手的同口径资料。"],
    )


def test_generate_report_with_pdf_and_sec_sources(
    tmp_path: Path,
) -> None:
    """简报应保存为独立文件并生成两种来源格式。"""
    retriever = FakeRetriever(build_results())
    service = ResearchReportService(
        retriever=retriever,
        output_dir=tmp_path,
        structured_model=FakeStructuredModel(build_draft()),
    )

    result = service.generate(
        question="Target有哪些经营重点和风险？",
        project_name="消费零售",
        report_date="2026-09-30",
        source=["target.pdf", "sec_target"],
    )

    assert result.output_path.is_file()
    assert result.output_path.read_text("utf-8") == result.markdown
    assert "PDF 第 3 页" in result.markdown
    assert "[SEC 官方原文](https://www.sec.gov/Archives/target.htm)" in result.markdown
    assert "缺少竞争对手的同口径资料" in result.markdown
    assert len(result.citations) == 2
    assert len(retriever.calls) == 2
    assert all(call["top_k"] == 4 for call in retriever.calls)
    assert retriever.calls[0]["source"] == ["target.pdf"]
    assert retriever.calls[1]["source"] == ["sec_target"]


def test_report_rejects_unknown_evidence_id(
    tmp_path: Path,
) -> None:
    """模型编造的证据 ID 不能写入最终简报。"""
    service = ResearchReportService(
        retriever=FakeRetriever(build_results()),
        output_dir=tmp_path,
        structured_model=FakeStructuredModel(
            build_draft("invented_id")
        ),
    )

    with pytest.raises(
        CitationValidationError,
        match="不存在的证据ID",
    ):
        service.generate(
            question="Target有哪些经营重点？",
            project_name="消费零售",
            report_date="2026-09-30",
            source=["target.pdf"],
        )

    assert list(tmp_path.iterdir()) == []


def test_report_moves_uncited_claim_to_gap(
    tmp_path: Path,
) -> None:
    """没有证据 ID 的陈述不进入事实章节，而是成为信息缺口。"""
    service = ResearchReportService(
        retriever=FakeRetriever(build_results()),
        output_dir=tmp_path,
        structured_model=FakeStructuredModel(build_draft("")),
    )

    result = service.generate(
        question="Target有哪些经营重点？",
        project_name="消费零售",
        report_date="2026-09-30",
        source=["target.pdf"],
    )

    assert "公司强调顾客体验" not in result.markdown
    assert "未找到足够证据支持“经营重点”" in result.markdown
