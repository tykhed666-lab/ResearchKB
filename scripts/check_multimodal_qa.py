"""验证多模态RAG回答是否正确引用图像证据。"""

from console_output import configure_utf8_stdout

from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.qa import RAGQuestionAnswerer
from research_kb.retrieval import MilvusRetriever
from research_kb.visual_evidence import (
    build_visual_evidence_id,
)

TEST_CASES = (
    (
        ("NVIDIA的AI五层架构从底层到顶层分别是什么，各层如何相互依赖？"),
        "01_NVDA_2026_Annual_Report.pdf",
        3,
    ),
    (
        (
            "Lumentum预计光学AI TAM从2025年的多少"
            "增长到2030年的多少，行业CAGR和"
            "2030年带宽结构分别是多少？"
        ),
        "03_LITE_OFC_2026_Investor_Briefing.pdf",
        30,
    ),
    (
        ("Seagate调查中，阻碍企业建设可持续数据存储的前三大因素和占比分别是什么？"),
        "05_STX_2025_Decarbonizing_Data_Report.pdf",
        11,
    ),
)

configure_utf8_stdout()


def main() -> None:
    """执行三次带图像证据的完整RAG问答。"""
    retriever = MilvusRetriever(
        store=MilvusStore(),
        embedding_service=EmbeddingService(),
    )

    answerer = RAGQuestionAnswerer(
        retriever=retriever,
        top_k=5,
    )

    passed_count = 0

    print(f"文本模型：{answerer.model_name}")

    for case_number, (
        question,
        expected_source,
        expected_page,
    ) in enumerate(
        TEST_CASES,
        start=1,
    ):
        expected_id = build_visual_evidence_id(
            source=expected_source,
            page_number=expected_page,
        )

        print(f"\n正在测试 {case_number}/3")
        print(f"问题：{question}")
        print(f"预期图像证据：{expected_id}")

        result = answerer.answer(question)

        cited_ids = {citation.evidence_id for citation in result.citations}

        image_cited = expected_id in cited_ids

        passed = not result.insufficient_information and image_cited

        if passed:
            passed_count += 1

        print(f"信息是否不足：{result.insufficient_information}")
        print(f"是否引用预期图像：{image_cited}")
        print(f"本题是否通过：{passed}")
        print(f"\n回答：\n{result.answer}")
        print("\n引用：")

        for citation in result.citations:
            print(
                f"- {citation.evidence_id} | "
                f"{citation.content_type} | "
                f"{citation.source} | "
                f"PDF第{citation.page_number}页 | "
                f"分数={citation.score:.4f}"
            )

    print("\n多模态问答验收：")
    print(f"通过数量：{passed_count}/{len(TEST_CASES)}")


if __name__ == "__main__":
    main()
