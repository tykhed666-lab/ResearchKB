"""管理Milvus连接和Collection结构。"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
VECTOR_DIMENSION = 1024


class MilvusStore:
    """管理ResearchKB使用的Milvus Collection。"""

    def __init__(self) -> None:
        """读取配置并连接Milvus。"""
        load_dotenv(ENV_PATH)

        milvus_uri = os.getenv("MILVUS_URI")
        collection_name = os.getenv(
            "MILVUS_COLLECTION",
            "research_knowledge",
        )

        if not milvus_uri:
            raise ValueError("缺少环境变量：MILVUS_URI")

        self.uri = milvus_uri
        self.collection_name = collection_name
        # 创建一个 Milvus 数据库的客户端连接对象，保存到实例属性上
        self.client = MilvusClient(uri=milvus_uri)

    def ensure_collection(self) -> bool:
        """确保项目需要的Collection存在。

        Returns:
            如果本次创建了Collection，返回True；
            如果Collection原本已经存在，返回False。
        """
        if self.client.has_collection(self.collection_name):
            return False

        schema = MilvusClient.create_schema(
            auto_id=False,
            # 关闭动态字段，防止拼错字段名后仍被写入。
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="evidence_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        # 文本的向量表示，用于相似度检索
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIMENSION,
        )
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="page_number",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="content_type",
            datatype=DataType.VARCHAR,
            max_length=32,
        )
        schema.add_field(
            field_name="document_hash",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        # 片段原文，检索到后直接展示给用户
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=4096,
        )

        # 初始化对象，用于后续添加索引
        index_params = MilvusClient.prepare_index_params()

        # 对向量字段添加索引
        # index_type="AUTOINDEX" 会根据数据自动选择最合适的索引类型
        # 用余弦相似度来衡量两个向量的接近程度
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # 创建 Collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Bounded",
        )

        return True

    def load_collection(self) -> None:
        """把Collection加载到内存，准备执行检索。"""
        self.client.load_collection(
            collection_name=self.collection_name,
        )

    def describe_collection(self) -> dict[str, Any]:
        """返回Collection的Schema信息。"""
        return self.client.describe_collection(
            collection_name=self.collection_name,
        )

    def list_indexes(self) -> list[str]:
        """返回Collection中的索引名称。"""
        return self.client.list_indexes(
            collection_name=self.collection_name,
        )

    def get_entity_count(self) -> int:
        """返回Collection中的实体数量。"""
        records = self.client.query(
            collection_name=self.collection_name,
            filter="",
            # count(*)只统计当前仍可查询的实体，
            # 不会把等待压缩的已删除数据计算在内。
            output_fields=["count(*)"],
        )

        if not records:
            return 0

        return int(records[0]["count(*)"])

    def get_existing_ids(
        self,
        evidence_ids: list[str],
    ) -> set[str]:
        """查询已经存在的证据ID。

        Args:
            evidence_ids: 准备入库的证据ID。

        Returns:
            已经存在于Milvus中的ID集合。
        """
        if not evidence_ids:
            return set()

        # 拿着一批 ID 去数据库里查，看哪些已经存在。
        records = self.client.query(
            collection_name=self.collection_name,
            ids=evidence_ids,
            output_fields=["evidence_id"],
        )

        return {
            record["evidence_id"]
            for record in records
        }

    def get_page_records(
        self,
        source: str,
        page_number: int,
        limit: int = 64,
    ) -> list[dict[str, Any]]:
        """读取同一 PDF 物理页的证据，供证据不足时补充上下文。

        Args:
            source: Milvus 中保存的 PDF 文件名。
            page_number: 用户看到的物理页码，从 1 开始。
            limit: 最多读取的片段数，避免意外读取过多内容。

        Returns:
            按页内片段序号排列的 Milvus 记录。
        """
        if not source.strip():
            raise ValueError("source 不能为空")

        if page_number <= 0 or limit <= 0:
            raise ValueError("页码和读取数量必须大于 0")

        # Milvus 的 filter 是字符串表达式，文件名中的特殊字符需要转义。
        safe_source = (
            self._escape_filter_text(source)
        )
        filter_expression = (
            f'source == "{safe_source}" '
            f"and page_number == {page_number}"
        )

        # vector 用于下一步计算补充片段与问题的真实相似度。
        records = self.client.query(
            collection_name=self.collection_name,
            filter=filter_expression,
            output_fields=[
                "evidence_id",
                "source",
                "page_number",
                "chunk_index",
                "content_type",
                "text",
                "vector",
            ],
            limit=limit,
        )

        return sorted(
            records,
            key=lambda record: (
                record["chunk_index"],
                record["content_type"],
            ),
        )

    def upsert_records(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """新增或更新一批Milvus实体。"""
        if not records:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            data=records,
        )

    def flush(self) -> None:
        """把已写入的数据持久化。"""
        self.client.flush(
            collection_name=self.collection_name,
        )

    @staticmethod
    def _escape_filter_text(
        value: str,
    ) -> str:
        """转义 Milvus 字符串过滤条件中的特殊字符。

        Milvus 的过滤条件最终是字符串表达式。
        如果文件名中包含反斜杠或双引号，
        必须先转义，避免表达式格式被破坏。
        """
        return (
            value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

    def get_source_statistics(
        self,
        source: str,
    ) -> dict[str, int]:
        """统计一份 PDF 在 Milvus 中的证据情况。

        Args:
            source:
                Milvus source 字段保存的 PDF 文件名。

        Returns:
            包含以下字段的字典：
            total：全部证据数量；
            text：文本证据数量；
            image：图像证据数量；
            max_page：证据覆盖到的最大 PDF 页码。
        """
        if not source.strip():
            raise ValueError(
                "source 不能为空"
            )

        safe_source = (
            self._escape_filter_text(source)
        )

        # 这里执行的是普通字段查询，
        # 不会调用 Embedding 模型，也不会产生模型费用。
        records = self.client.query(
            collection_name=self.collection_name,
            filter=(
                f'source == "{safe_source}"'
            ),
            output_fields=[
                "content_type",
                "page_number",
            ],
            limit=16384,
        )

        text_count = sum(
            record["content_type"] == "text"
            for record in records
        )

        image_count = sum(
            record["content_type"] == "image"
            for record in records
        )

        max_page = max(
            (
                int(record["page_number"])
                for record in records
            ),
            default=0,
        )

        return {
            "total": len(records),
            "text": text_count,
            "image": image_count,
            "max_page": max_page,
        }

    def delete_source_records(
        self,
        source: str,
        content_type: str | None = None,
    ) -> int:
        """删除一份PDF在Milvus中的证据。

        Args:
            source:
                Milvus source字段保存的PDF文件名。
            content_type:
                None表示删除全部证据；
                "text"只删除文本证据；
                "image"只删除图像证据。

        Returns:
            删除前匹配到的证据数量。
        """
        cleaned_source = source.strip()

        if not cleaned_source:
            raise ValueError(
                "source不能为空"
            )

        if content_type not in {
            None,
            "text",
            "image",
        }:
            raise ValueError(
                "content_type只能是"
                "text、image或None"
            )

        # 删除前先统计数量，使页面能够告诉用户
        # 本次实际清理了多少条证据。
        statistics = (
            self.get_source_statistics(
                cleaned_source
            )
        )

        if content_type is None:
            deleted_count = (
                statistics["total"]
            )
        else:
            deleted_count = (
                statistics[content_type]
            )

        if deleted_count == 0:
            return 0

        safe_source = (
            self._escape_filter_text(
                cleaned_source
            )
        )

        filter_expression = (
            f'source == "{safe_source}"'
        )

        if content_type is not None:
            filter_expression += (
                " and content_type == "
                f'"{content_type}"'
            )

        # delete只处理Milvus中的向量和证据字段，
        # 不会删除SQLite记录或本地PDF。
        self.client.delete(
            collection_name=(
                self.collection_name
            ),
            filter=filter_expression,
        )

        # 立即持久化删除结果，供后续重新入库使用。
        self.flush()

        return deleted_count

    def list_sources(self) -> list[str]:
        """返回Collection中所有不重复的PDF文件名。"""
        records = self.client.query(
            collection_name=self.collection_name,
            filter="",
            limit=16384,
            output_fields=["source"],
        )

        return sorted(
            {
                record["source"]
                for record in records
            }
        )
