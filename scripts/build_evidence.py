"""生成首轮 100 页 PDF 的证据 JSON。"""

import json
from dataclasses import dataclass
from pathlib import Path

from research_kb.chunker import EvidenceChunk, split_pages_into_chunks
from research_kb.pdf_loader import load_pdf_pages
from research_kb.text_cleaner import clean_pdf_pages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "first_pass_evidence.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


@dataclass(frozen=True, slots=True)
class DocumentPlan:
    """表示一份 PDF 的首轮处理计划。

    Attributes:
        filename: PDF 文件名。
        page_limit: 从开头处理的物理页数。
    """

    filename: str
    page_limit: int


DOCUMENT_PLANS = (
    DocumentPlan(
        filename="01_NVDA_2026_Annual_Report.pdf",
        page_limit=20,
    ),
    DocumentPlan(
        filename="02_AMD_2025_Annual_Report.pdf",
        page_limit=7,
    ),
    DocumentPlan(
        filename="03_LITE_OFC_2026_Investor_Briefing.pdf",
        page_limit=30,
    ),
    DocumentPlan(
        filename="04_MU_2025_Annual_Report.pdf",
        page_limit=15,
    ),
    DocumentPlan(
        filename="05_STX_2025_Decarbonizing_Data_Report.pdf",
        page_limit=28,
    ),
)


def process_document(
    plan: DocumentPlan,
) -> tuple[list[EvidenceChunk], int]:
    """按照计划加载、清理并切分一份 PDF。

    Args:
        plan: 当前 PDF 的文件名和处理页数。

    Returns:
        证据片段列表和实际处理的页数。

    Raises:
        ValueError: PDF 的实际页数少于计划页数。
    """
    pdf_path = RAW_DATA_DIR / plan.filename
    raw_pages = load_pdf_pages(pdf_path)

    selected_pages = raw_pages[: plan.page_limit]

    if len(selected_pages) != plan.page_limit:
        raise ValueError(
            f"{plan.filename} 计划处理 {plan.page_limit} 页，"
            f"但实际只能读取 {len(selected_pages)} 页"
        )

    cleaned_pages = clean_pdf_pages(selected_pages)

    evidence_chunks = split_pages_into_chunks(
        cleaned_pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return evidence_chunks, len(cleaned_pages)


def validate_chunks(chunks: list[EvidenceChunk]) -> None:
    """检查证据 ID 和片段长度是否符合要求。

    Args:
        chunks: 所有文档生成的证据片段。

    Raises:
        ValueError: 没有生成证据、ID 重复或片段过长。
    """
    if not chunks:
        raise ValueError("没有生成任何证据片段")

    evidence_ids = [chunk.evidence_id for chunk in chunks]

    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("发现重复的 evidence_id")

    oversized_chunks = [chunk for chunk in chunks if len(chunk.text) > CHUNK_SIZE]

    if oversized_chunks:
        raise ValueError(f"发现 {len(oversized_chunks)} 个过长证据片段")


def main() -> None:
    """处理首轮 100 页资料并保存证据 JSON。"""
    all_chunks: list[EvidenceChunk] = []
    inspection_samples: list[EvidenceChunk] = []
    total_pages = 0

    print("开始处理首轮 PDF：")

    for plan in DOCUMENT_PLANS:
        chunks, processed_pages = process_document(plan)

        all_chunks.extend(chunks)
        total_pages += processed_pages

        # 每份文档保留前两条，共得到 10 条人工抽查样本。
        inspection_samples.extend(chunks[:2])

        print(f"- {plan.filename}：{processed_pages} 页，{len(chunks)} 条证据")

    validate_chunks(all_chunks)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_data = [chunk.to_dict() for chunk in all_chunks]

    OUTPUT_PATH.write_text(
        json.dumps(
            evidence_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    max_length = max(len(chunk.text) for chunk in all_chunks)

    print("\n处理完成：")
    print(f"总页数：{total_pages}")
    print(f"证据总数：{len(all_chunks)}")
    print(f"唯一 ID 数：{len(set(chunk.evidence_id for chunk in all_chunks))}")
    print(f"最长片段：{max_length}")
    print(f"输出位置：{OUTPUT_PATH}")

    print("\n10 条人工抽查样本：")

    for chunk in inspection_samples:
        print(
            f"\n{chunk.evidence_id} | "
            f"{chunk.source} | "
            f"第 {chunk.page_number} 页 | "
            f"页内第 {chunk.chunk_index} 段"
        )
        print(chunk.text[:200])


if __name__ == "__main__":
    main()
