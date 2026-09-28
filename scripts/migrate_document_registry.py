"""把已有 PDF 和 Milvus 证据迁移到 SQLite 注册表。"""

from pathlib import Path

import pymupdf

from research_kb.document_registry import (
    DocumentMetadata,
    DocumentRegistry,
)
from research_kb.indexer import (
    calculate_document_hash,
)
from research_kb.milvus_store import (
    MilvusStore,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw"
)


# 历史资料没有经过上传页面，
# 因此在迁移脚本中补充已知的业务元数据。
#
# 没有可靠核验的报告日期暂时保存为 None，
# 避免为了填满字段而写入猜测日期。
KNOWN_METADATA = {
    "01_NVDA_2026_Annual_Report.pdf": (
        DocumentMetadata(
            title=(
                "NVIDIA 2026 Annual Report"
            ),
            document_type="年报",
            company="NVIDIA",
            ticker="NVDA",
            industry="半导体",
            report_date=None,
        )
    ),
    "02_AMD_2025_Annual_Report.pdf": (
        DocumentMetadata(
            title=(
                "AMD 2025 Annual Report"
            ),
            document_type="年报",
            company="AMD",
            ticker="AMD",
            industry="半导体",
            report_date=None,
        )
    ),
    (
        "03_LITE_OFC_2026_"
        "Investor_Briefing.pdf"
    ): (
        DocumentMetadata(
            title=(
                "Lumentum OFC 2026 "
                "Investor Briefing"
            ),
            document_type="投资者演示",
            company="Lumentum",
            ticker="LITE",
            industry="光模块",
            report_date="2026-03-17",
        )
    ),
    "04_MU_2025_Annual_Report.pdf": (
        DocumentMetadata(
            title=(
                "Micron 2025 Annual Report"
            ),
            document_type="年报",
            company="Micron",
            ticker="MU",
            industry="存储",
            report_date=None,
        )
    ),
    (
        "05_STX_2025_"
        "Decarbonizing_Data_Report.pdf"
    ): (
        DocumentMetadata(
            title="Decarbonizing Data",
            document_type="行业报告",
            company="Seagate",
            ticker="STX",
            industry="数据存储",
            report_date="2025-04-01",
        )
    ),
    "06_TGT_2025_Annual_Report.pdf": (
        DocumentMetadata(
            title=(
                "Target 2025 Annual Report"
            ),
            document_type="年报",
            company="Target",
            ticker="TGT",
            industry="消费零售",
            report_date=None,
        )
    ),
}


def read_total_pages(
    pdf_path: Path,
) -> int:
    """读取 PDF 总页数，并确保文件可正常打开。"""
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError(
                f"PDF 需要密码：{pdf_path.name}"
            )

        return document.page_count


def main() -> None:
    """迁移已有资料，重复执行时跳过相同哈希。"""
    registry = DocumentRegistry()

    store = MilvusStore()
    store.ensure_collection()
    store.load_collection()

    added_count = 0
    skipped_count = 0
    failed_count = 0

    pdf_paths = sorted(
        RAW_DATA_DIR.glob("*.pdf")
    )

    print(
        f"发现本地 PDF：{len(pdf_paths)}"
    )

    for pdf_path in pdf_paths:
        document_hash = (
            calculate_document_hash(
                pdf_path
            )
        )

        existing = registry.get_by_hash(
            document_hash
        )

        if existing is not None:
            print(
                f"跳过：{pdf_path.name} "
                f"| 状态={existing.status}"
            )
            skipped_count += 1
            continue

        total_pages = read_total_pages(
            pdf_path
        )

        statistics = (
            store.get_source_statistics(
                pdf_path.name
            )
        )

        metadata = KNOWN_METADATA.get(
            pdf_path.name,
            DocumentMetadata(
                title=pdf_path.stem,
                document_type="未分类",
            ),
        )

        record, _ = (
            registry.register_document(
                document_hash=document_hash,
                original_name=pdf_path.name,
                saved_name=pdf_path.name,
                metadata=metadata,
                total_pages=total_pages,
            )
        )

        if statistics["total"] > 0:
            registry.mark_indexed(
                document_id=(
                    record.document_id
                ),
                total_pages=total_pages,
                processed_pages=(
                    statistics["max_page"]
                ),
                text_chunk_count=(
                    statistics["text"]
                ),
                image_chunk_count=(
                    statistics["image"]
                ),
            )

            print(
                f"新增：{pdf_path.name}"
            )
            print(
                f"  总页数：{total_pages}"
            )
            print(
                "  已处理到："
                f"{statistics['max_page']}页"
            )
            print(
                "  文本证据："
                f"{statistics['text']}"
            )
            print(
                "  图像证据："
                f"{statistics['image']}"
            )

            added_count += 1

        else:
            registry.mark_failed(
                document_id=(
                    record.document_id
                ),
                error_message=(
                    "Milvus 中没有找到"
                    "该文件的证据"
                ),
            )

            print(
                f"失败：{pdf_path.name}"
                "，Milvus 中没有证据"
            )

            failed_count += 1

    documents = registry.list_documents()

    print()
    print("迁移完成：")
    print(f"新增文档：{added_count}")
    print(f"跳过文档：{skipped_count}")
    print(f"失败文档：{failed_count}")
    print(
        f"注册表文档数：{len(documents)}"
    )
    print(
        "Milvus实体数："
        f"{store.get_entity_count()}"
    )

    print()
    print("SQLite 文档目录：")

    for document in documents:
        print(
            f"- {document.ticker or '无Ticker'}"
            f" | {document.saved_name}"
            f" | 状态={document.status}"
            f" | 文本={document.text_chunk_count}"
            f" | 图像={document.image_chunk_count}"
        )


if __name__ == "__main__":
    main()