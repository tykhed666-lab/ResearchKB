"""验收5个库内问题和3个库外问题。"""

import json
from dataclasses import dataclass
from pathlib import Path

from console_output import configure_utf8_stdout

from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.qa import RAGQuestionAnswerer
from research_kb.retrieval import MilvusRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "qa_evaluation.json"

configure_utf8_stdout()


@dataclass(frozen=True, slots=True)
class QATestCase:
    """表示一个RAG验收问题。"""

    question: str
    should_answer: bool
    expected_source: str | None = None


TEST_CASES = (
    QATestCase(
        question="AMD 2025年数据中心收入是多少，较上年增长多少？",
        should_answer=True,
        expected_source="02_AMD_2025_Annual_Report.pdf",
    ),
    QATestCase(
        question="NVIDIA如何描述AI基础设施的主要层级？",
        should_answer=True,
        expected_source="01_NVDA_2026_Annual_Report.pdf",
    ),
    QATestCase(
        question="Lumentum预计OCS业务何时达到怎样的收入运行率？",
        should_answer=True,
        expected_source="03_LITE_OFC_2026_Investor_Briefing.pdf",
    ),
    QATestCase(
        question="Micron的HBM、DRAM与NAND在AI数据中心中承担什么作用？",
        should_answer=True,
        expected_source="04_MU_2025_Annual_Report.pdf",
    ),
    QATestCase(
        question="Seagate报告列出的数据中心扩容选择和主要限制是什么？",
        should_answer=True,
        expected_source="05_STX_2025_Decarbonizing_Data_Report.pdf",
    ),
    QATestCase(
        question="苹果公司2025财年的iPhone收入是多少？",
        should_answer=False,
    ),
    QATestCase(
        question="特斯拉2030年的汽车交付目标是多少？",
        should_answer=False,
    ),
    QATestCase(
        question="2026年世界杯冠军是谁？",
        should_answer=False,
    ),
)


def evaluate_result(
    case: QATestCase,
    insufficient_information: bool,
    citation_sources: set[str],
) -> bool:
    """根据问题类型判断结果是否通过。

    库内问题必须正常回答、有引用，并命中预期资料。
    库外问题必须明确标记信息不足。
    """
    if case.should_answer:
        return not insufficient_information and case.expected_source in citation_sources

    return insufficient_information


def main() -> None:
    """运行8个问题并保存完整问答记录。"""
    retriever = MilvusRetriever(
        store=MilvusStore(),
        embedding_service=EmbeddingService(),
    )

    answerer = RAGQuestionAnswerer(
        retriever=retriever,
        top_k=5,
    )

    evaluation_records: list[dict] = []

    answerable_total = 0
    answerable_passed = 0
    unanswerable_total = 0
    unanswerable_passed = 0

    for case_number, case in enumerate(
        TEST_CASES,
        start=1,
    ):
        print(f"\n正在测试 {case_number}/{len(TEST_CASES)}")
        print(f"问题：{case.question}")

        result = answerer.answer(case.question)

        citation_sources = {citation.source for citation in result.citations}

        passed = evaluate_result(
            case=case,
            insufficient_information=(result.insufficient_information),
            citation_sources=citation_sources,
        )

        if case.should_answer:
            answerable_total += 1
            if passed:
                answerable_passed += 1
        else:
            unanswerable_total += 1
            if passed:
                unanswerable_passed += 1

        citation_records = [
            {
                "evidence_id": citation.evidence_id,
                "source": citation.source,
                "page_number": citation.page_number,
                "score": citation.score,
            }
            for citation in result.citations
        ]

        evaluation_records.append(
            {
                "question": case.question,
                "should_answer": case.should_answer,
                "expected_source": case.expected_source,
                "passed": passed,
                "answer": result.answer,
                "insufficient_information": (result.insufficient_information),
                "missing_information": (result.missing_information),
                "retrieved_count": result.retrieved_count,
                "citations": citation_records,
            }
        )

        print(f"应当回答：{case.should_answer}")
        print(f"信息是否不足：{result.insufficient_information}")
        print(f"本题是否通过：{passed}")
        print(f"回答：{result.answer}")

        print("引用：")

        if not result.citations:
            print("- 无")
        else:
            for citation in result.citations:
                print(f"- [{citation.evidence_id}] {citation.citation}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "model": answerer.model_name,
        "answerable": {
            "passed": answerable_passed,
            "total": answerable_total,
        },
        "unanswerable": {
            "passed": unanswerable_passed,
            "total": unanswerable_total,
        },
        "records": evaluation_records,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n最终验收：")
    print(f"库内问题：{answerable_passed}/{answerable_total}")
    print(f"库外拒答：{unanswerable_passed}/{unanswerable_total}")
    print(f"记录文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
