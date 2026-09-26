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
        stats = self.client.get_collection_stats(
            collection_name=self.collection_name,
        )

        return int(stats["row_count"])

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

        records = self.client.query(
            collection_name=self.collection_name,
            ids=evidence_ids,
            output_fields=["evidence_id"],
        )

        return {
            record["evidence_id"]
            for record in records
        }

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