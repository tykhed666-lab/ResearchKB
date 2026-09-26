"""检查查询和文档Embedding是否能够正常生成。"""

from research_kb.embedding_service import EmbeddingService


def main() -> None:
    """生成少量测试向量并打印结构信息。"""
    service = EmbeddingService()

    sample_documents = [
        "NVIDIA describes AI infrastructure as a multi-layer technology stack.",
        "HBM provides high-bandwidth memory for AI accelerators.",
    ]

    query = "What role does memory play in AI infrastructure?"

    document_vectors = service.embed_documents(sample_documents)
    query_vector = service.embed_query(query)

    print(f"模型：{service.model_name}")
    print(f"文档数量：{len(sample_documents)}")
    print(f"文档向量数量：{len(document_vectors)}")
    print(f"文档向量维度：{len(document_vectors[0])}")
    print(f"查询向量维度：{len(query_vector)}")

    # 遍历所有文档向量，检查维度是否正确
    # 如果全部正确，返回 True，否则返回 False
    dimensions_are_valid = all(
        len(vector) == service.dimension
        for vector in document_vectors
    )

    print(f"全部文档向量维度正确：{dimensions_are_valid}")


if __name__ == "__main__":
    main()