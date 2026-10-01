"""创建并检查ResearchKB使用的Milvus Collection。"""

from research_kb.milvus_store import MilvusStore


def main() -> None:
    """创建Collection并显示Schema。"""
    store = MilvusStore()

    created = store.ensure_collection()
    store.load_collection()

    description = store.describe_collection()
    field_names = [field["name"] for field in description["fields"]]

    print(f"Milvus地址：{store.uri}")
    print(f"Collection：{store.collection_name}")
    print(f"本次是否新建：{created}")
    print(f"字段：{field_names}")
    print(f"索引：{store.list_indexes()}")


if __name__ == "__main__":
    main()
