"""封装项目使用的文本 Embedding 模型。"""

from langchain_openai import OpenAIEmbeddings

from research_kb.settings import require_environment

EXPECTED_DIMENSION = 1024


class EmbeddingService:
    """为文档和查询文本生成向量。"""

    def __init__(self) -> None:
        """读取环境变量并初始化 LangChain Embedding 模型。"""
        api_key, base_url, model_name = require_environment(
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "EMBEDDING_MODEL",
        )

        self.model_name = model_name
        # dimension是预期的向量维度。
        self.dimension = EXPECTED_DIMENSION

        self._model = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            # 每批发送较少文本，避免一次请求过大。
            chunk_size=10,
            max_retries=2,
            timeout=60,
            # 使用兼容接口时，不让 LangChain 使用 OpenAI 专用分词器。
            check_embedding_ctx_length=False,
        )

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 10,
    ) -> list[list[float]]:
        """批量生成文档向量。

        Args:
            texts: 需要向量化的文档文本。
            batch_size: 每次API请求处理的文本数量。

        Returns:
            与输入文本顺序一致的向量列表。

        Raises:
            ValueError: 输入为空、包含空文本或批次大小不正确。
        """
        if not texts:
            return []

        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if any(not text.strip() for text in texts):
            raise ValueError("Embedding 输入中不能包含空文本")

        vectors: list[list[float]] = []

        for start_index in range(0, len(texts), batch_size):
            batch = texts[start_index : start_index + batch_size]
            batch_vectors = self._model.embed_documents(batch)
            vectors.extend(batch_vectors)

        self._validate_vectors(vectors)

        return vectors

    def embed_query(self, query: str) -> list[float]:
        """为用户查询生成一个向量。

        Args:
            query: 用户查询文本。

        Returns:
            1024维查询向量。

        Raises:
            ValueError: 查询为空或返回维度不正确。
        """
        if not query.strip():
            raise ValueError("查询文本不能为空")
        # 本质是把用户的查询文本发送给 Embedding 模型。
        # 返回一个1024维的向量
        vector = self._model.embed_query(query)
        self._validate_vectors([vector])

        return vector

    def _validate_vectors(
        self,
        vectors: list[list[float]],
    ) -> None:
        """检查模型返回的向量维度。"""
        invalid_dimensions = [
            len(vector) for vector in vectors if len(vector) != self.dimension
        ]

        if invalid_dimensions:
            raise ValueError(
                f"期望向量维度为 {self.dimension}，但发现异常维度：{invalid_dimensions}"
            )
