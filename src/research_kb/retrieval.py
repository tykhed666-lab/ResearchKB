"""使用Embedding和Milvus执行证据检索。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.settings import DEFAULT_TOP_K


@dataclass(frozen=True, slots=True)
class RetrievalSourceMetadata:
    """为 Milvus 结果补充不适合放入向量表的来源信息。"""

    source_type: str
    display_title: str
    source_url: str | None = None
    source_date: str | None = None


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
    source_type: str = "pdf"
    display_title: str | None = None
    source_url: str | None = None
    source_date: str | None = None

    @property
    def citation(self) -> str:
        """生成便于展示的来源引用。"""
        title = self.display_title or self.source
        if self.source_type == "sec":
            return f"{title}，SEC 官方原文"
        return f"{title}，PDF第{self.page_number}页"


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
        top_k: int = DEFAULT_TOP_K,
        source: str | Sequence[str] | None = None,
        source_metadata: (
            Mapping[str, RetrievalSourceMetadata]
            | None
        ) = None,
    ) -> list[RetrievalResult]:
        """执行COSINE向量检索。

        Args:
            query: 用户问题。
            top_k: 返回的最大证据数量。
            source:
                可选的单个 PDF 文件名，
                或多个 PDF 文件名组成的序列。

        Returns:
            按相似度从高到低排列的检索结果。

        Raises:
            ValueError: 问题为空或top_k不正确。
        """
        if not query.strip():
            raise ValueError("检索问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k必须大于0")

        filter_expression = ""

        if source is not None:
            # 字符串本身也是 Sequence，
            # isinstance 是 Python 的内置函数，用来判断"一个对象是不是某种类型
            # 所以必须先单独判断 str。
            if isinstance(source, str):
                source_names = [source]
            else:
                source_names = list(source)

            # 空列表表示页面没有选中任何文档。
            # 此时必须返回空结果，不能错误地搜索全库。
            if not source_names:
                return []

            cleaned_source_names: list[str] = []

            for source_name in source_names:
                cleaned_name = (
                    source_name.strip()
                )

                if not cleaned_name:
                    raise ValueError(
                        "source 过滤条件"
                        "不能包含空字符串"
                    )

                cleaned_source_names.append(
                    cleaned_name
                )

            # dict.fromkeys 保留原顺序并去除重复文件名。
            unique_source_names = list(
                dict.fromkeys(
                    cleaned_source_names
                )
            )

            escaped_source_names = [
                source_name
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                for source_name
                in unique_source_names
            ]

            if len(escaped_source_names) == 1:
                filter_expression = (
                    'source == '
                    f'"{escaped_source_names[0]}"'
                )

            else:
                # Milvus 的 in 表达式允许一次限定
                # 多个 source 字段值。
                quoted_sources = ", ".join(
                    f'"{source_name}"'
                    for source_name
                    in escaped_source_names
                )

                filter_expression = (
                    f"source in "
                    f"[{quoted_sources}]"
                )

        # 先完成来源校验。空来源列表会在上方直接返回，
        # 因此不会产生一次无意义的 Embedding 调用。
        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

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
            source_name = entity["source"]
            metadata = (
                source_metadata.get(source_name)
                if source_metadata is not None
                else None
            )

            results.append(
                RetrievalResult(
                    evidence_id=hit["evidence_id"],
                    score=float(hit["distance"]),
                    source=source_name,
                    page_number=entity["page_number"],
                    chunk_index=entity["chunk_index"],
                    content_type=entity["content_type"],
                    text=entity["text"],
                    source_type=(
                        metadata.source_type
                        if metadata is not None
                        else "pdf"
                    ),
                    display_title=(
                        metadata.display_title
                        if metadata is not None
                        else source_name
                    ),
                    source_url=(
                        metadata.source_url
                        if metadata is not None
                        else None
                    ),
                    source_date=(
                        metadata.source_date
                        if metadata is not None
                        else None
                    ),
                )
            )

        return results

    def expand_top_result_page(
        self,
        query: str,
        results: list[RetrievalResult],
        max_page_chunks: int = 16,
    ) -> list[RetrievalResult]:
        """补充首条检索结果所在物理页的其他证据片段。

        复杂排版的 PDF 页面可能被切成多个片段，而 Top-K 只召回
        其中一部分。这个方法在首次回答证据不足时，读取同页片段，
        让模型能够恢复该页更完整的上下文。

        Args:
            query: 用户原始问题。
            results: 第一次向量检索返回的证据。
            max_page_chunks: 最多补充的同页片段数，避免上下文过长。

        Returns:
            同页片段加上原有其他检索结果，并去除重复证据。
        """
        if not query.strip():
            raise ValueError("检索问题不能为空")

        if max_page_chunks <= 0:
            raise ValueError("max_page_chunks 必须大于 0")

        if not results:
            return []

        # 第一条结果相似度最高，用它的来源和页码确定需要补充的页面。
        top_result = results[0]
        # SEC 网页没有物理页码，不能套用 PDF 的同页补证据逻辑。
        if (
            top_result.source_type != "pdf"
            or top_result.page_number <= 0
        ):
            return results
        # 获取同页的其他片段
        page_records = self.store.get_page_records(
            source=top_result.source,
            page_number=top_result.page_number,
            limit=max_page_chunks,
        )

        if not page_records:
            return results

        # Milvus 精确查询不会返回相似度，因此在这里重新计算。
        query_vector = self.embedding_service.embed_query(query)

        page_results = [
            RetrievalResult(
                evidence_id=record["evidence_id"],
                score=self._cosine_similarity(
                    query_vector,
                    record["vector"],
                ),
                source=record["source"],
                page_number=record["page_number"],
                chunk_index=record["chunk_index"],
                content_type=record["content_type"],
                text=record["text"],
                source_type=top_result.source_type,
                display_title=top_result.display_title,
                source_url=top_result.source_url,
                source_date=top_result.source_date,
            )
            for record in page_records
        ]

        page_evidence_ids = {
            result.evidence_id
            for result in page_results
        }

        # 同页片段按照页面顺序放在前面；原检索结果中属于其他页面的
        # 证据继续保留，避免丢失跨页信息。
        other_results = [
            result
            for result in results
            if result.evidence_id not in page_evidence_ids
        ]

        return [*page_results, *other_results]

    @staticmethod
    def _cosine_similarity(
        first_vector: list[float],
        second_vector: list[float],
    ) -> float:
        """计算两个相同维度向量的余弦相似度。"""
        if len(first_vector) != len(second_vector):
            raise ValueError("计算相似度的向量维度不一致")

        dot_product = sum(
            first_value * second_value
            for first_value, second_value in zip(
                first_vector,
                second_vector,
                strict=True,
            )
        )

        first_norm = sqrt(
            sum(value * value for value in first_vector)
        )
        second_norm = sqrt(
            sum(value * value for value in second_vector)
        )

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return dot_product / (first_norm * second_norm)
