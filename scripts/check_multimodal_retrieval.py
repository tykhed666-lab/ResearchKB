"""验证多模态图像证据的Top-5召回效果。"""

from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.retrieval import MilvusRetriever
from research_kb.visual_evidence import (
    build_visual_evidence_id,
)


TEST_CASES = (
    (
        (
            "NVIDIA的AI五层架构从底层到顶层"
            "分别是什么，各层如何相互依赖？"
        ),
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
        (
            "Seagate调查中，阻碍企业建设可持续"
            "数据存储的前三大因素和占比分别是什么？"
        ),
        "05_STX_2025_Decarbonizing_Data_Report.pdf",
        11,
    ),
)


def main() -> None:
    """运行三个多模态Top-5检索测试。"""
    retriever = MilvusRetriever(
        store=MilvusStore(),
        embedding_service=EmbeddingService(),
    )

    passed_count = 0

    for case_number, (
        query,
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

        results = retriever.search(
            query=query,
            top_k=5,
        )

        matched_rank = None

        for rank, result in enumerate(
            results,
            start=1,
        ):
            if (
                result.evidence_id == expected_id
                and result.content_type == "image"
            ):
                matched_rank = rank
                break

        passed = matched_rank is not None

        if passed:
            passed_count += 1

        print(f"\n测试{case_number}")
        print(f"问题：{query}")
        print(f"预期图像证据：{expected_id}")
        print(f"Top-5命中：{passed}")
        print(f"命中排名：{matched_rank}")

        for rank, result in enumerate(
            results,
            start=1,
        ):
            evidence_type = (
                "图像"
                if result.content_type == "image"
                else "文本"
            )

            preview = " ".join(
                result.text.split()
            )[:120]

            print(
                f"  {rank}. "
                f"{evidence_type} | "
                f"分数={result.score:.4f} | "
                f"{result.evidence_id} | "
                f"{result.source} | "
                f"PDF第{result.page_number}页"
            )
            print(f"     {preview}")

    print("\n多模态检索验收：")
    print(
        f"通过数量："
        f"{passed_count}/{len(TEST_CASES)}"
    )


if __name__ == "__main__":
    main()