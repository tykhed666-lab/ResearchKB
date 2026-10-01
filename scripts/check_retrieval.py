"""使用5个已知来源的问题验收Milvus检索。"""

from console_output import configure_utf8_stdout

from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.retrieval import MilvusRetriever

TEST_CASES = (
    (
        "AMD 2025年数据中心收入是多少，较上年增长多少？",
        "02_AMD_2025_Annual_Report.pdf",
    ),
    (
        "NVIDIA如何描述AI基础设施的主要层级？",
        "01_NVDA_2026_Annual_Report.pdf",
    ),
    (
        "Lumentum预计OCS业务何时达到怎样的收入运行率？",
        "03_LITE_OFC_2026_Investor_Briefing.pdf",
    ),
    (
        "Micron的HBM、DRAM与NAND在AI数据中心中承担什么作用？",
        "04_MU_2025_Annual_Report.pdf",
    ),
    (
        "Seagate报告列出的数据中心扩容选择和主要限制是什么？",
        "05_STX_2025_Decarbonizing_Data_Report.pdf",
    ),
)

configure_utf8_stdout()


def main() -> None:
    """执行5个Top-5检索测试和一次文件过滤测试。"""
    retriever = MilvusRetriever(
        store=MilvusStore(),
        embedding_service=EmbeddingService(),
    )

    passed_count = 0

    for case_number, (query, expected_source) in enumerate(
        TEST_CASES,
        start=1,
    ):
        results = retriever.search(
            query=query,
            top_k=5,
        )

        retrieved_sources = {result.source for result in results}

        passed = expected_source in retrieved_sources

        if passed:
            passed_count += 1

        print(f"\n问题{case_number}：{query}")
        print(f"预期来源：{expected_source}")
        print(f"Top-5包含预期来源：{passed}")

        for rank, result in enumerate(results, start=1):
            preview = " ".join(result.text.split())[:160]

            print(f"  {rank}. 分数={result.score:.4f} | {result.citation}")
            print(f"     {preview}")

    print("\n检索验收：")
    print(f"通过数量：{passed_count}/{len(TEST_CASES)}")

    # 单文件过滤测试：所有结果必须来自AMD报告。
    filter_source = "02_AMD_2025_Annual_Report.pdf"

    filtered_results = retriever.search(
        query=TEST_CASES[0][0],
        top_k=3,
        source=filter_source,
    )

    filter_passed = len(filtered_results) == 3 and all(
        result.source == filter_source for result in filtered_results
    )

    print(f"单文件过滤：{filter_passed}")


if __name__ == "__main__":
    main()
