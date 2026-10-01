"""运行一次带引用的RAG问答。"""

from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.qa import RAGQuestionAnswerer
from research_kb.retrieval import MilvusRetriever

QUESTION = "NVIDIA如何描述AI基础设施的主要层级？"


def main() -> None:
    """执行一次完整的检索增强问答。"""
    retriever = MilvusRetriever(
        store=MilvusStore(),
        embedding_service=EmbeddingService(),
    )

    answerer = RAGQuestionAnswerer(
        retriever=retriever,
        top_k=5,
    )

    result = answerer.answer(QUESTION)

    print(f"模型：{answerer.model_name}")
    print(f"问题：{result.question}")
    print(f"检索证据数：{result.retrieved_count}")
    print(f"信息是否不足：{result.insufficient_information}")
    print(f"\n回答：\n{result.answer}")

    if result.missing_information:
        print(f"\n缺少的信息：{result.missing_information}")

    print("\n引用：")

    if not result.citations:
        print("- 无")
        return

    for citation in result.citations:
        print(
            f"- [{citation.evidence_id}] "
            f"{citation.citation}，"
            f"相似度={citation.score:.4f}"
        )


if __name__ == "__main__":
    main()
