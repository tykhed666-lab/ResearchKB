"""使用Embedding和Milvus执行证据检索。"""

from dataclasses import dataclass

from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """表示一条Milvus检索结果。"""

    evidence_id: str
    score: float
    source: str
    page_number: int
    chunk_index: int
    content_type: str
    text: str

    @property
    def citation(self) -> str:
        """生成便于展示的来源引用。"""
        return f"{self.source}，PDF第{self.page_number}页"


class MilvusRetriever:
    """把用户问题转换成向量并检索Milvus。"""

    def __init__(
        self,
        store: MilvusStore,
        embedding_service: EmbeddingService,
    ) -> None:
        """保存Milvus和Embedding服务并加载Collection。"""
        self.store = store
        self.embedding_service = embedding_service

        self.store.ensure_collection()
        self.store.load_collection()

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        """执行COSINE向量检索。

        Args:
            query: 用户问题。
            top_k: 返回的最大证据数量。
            source: 可选的PDF文件名过滤条件。

        Returns:
            按相似度从高到低排列的检索结果。

        Raises:
            ValueError: 问题为空或top_k不正确。
        """
        if not query.strip():
            raise ValueError("检索问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k必须大于0")

        query_vector = self.embedding_service.embed_query(query)

        filter_expression = ""

        if source is not None:
            if not source.strip():
                raise ValueError("source过滤条件不能为空字符串")

            # 转义过滤表达式中的反斜杠和双引号。
            safe_source = (
                source
                .replace("\\", "\\\\")
                .replace('"', '\\"')
            )
            filter_expression = f'source == "{safe_source}"'

        search_response = self.store.client.search(
            collection_name=self.store.collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=top_k,
            filter=filter_expression,
            output_fields=[
                "source",
                "page_number",
                "chunk_index",
                "content_type",
                "text",
            ],
            search_params={
                "metric_type": "COSINE",
                "params": {},
            },
        )

        if not search_response:
            return []

        results: list[RetrievalResult] = []

        for hit in search_response[0]:
            entity = hit["entity"]

            results.append(
                RetrievalResult(
                    evidence_id=hit["evidence_id"],
                    score=float(hit["distance"]),
                    source=entity["source"],
                    page_number=entity["page_number"],
                    chunk_index=entity["chunk_index"],
                    content_type=entity["content_type"],
                    text=entity["text"],
                )
            )

        return results