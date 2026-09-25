"""生成并检查 NVIDIA 年报前 10 页的证据 JSON。"""

import json
from pathlib import Path

from research_kb.chunker import split_pages_into_chunks
from research_kb.pdf_loader import load_pdf_pages
from research_kb.text_cleaner import clean_pdf_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "01_NVDA_2026_Annual_Report.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nvidia_first_10_pages.json"
)


def main() -> None:
    """读取、清理、切分 PDF，并把证据保存为 JSON。"""
    raw_pages = load_pdf_pages(PDF_PATH)

    # 首次实验只处理前 10 页，便于快速检查结果。
    selected_pages = raw_pages[:10]
    cleaned_pages = clean_pdf_pages(selected_pages)

    evidence_chunks = split_pages_into_chunks(
        cleaned_pages,
        chunk_size=800,
        chunk_overlap=120,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    evidence_data = [
        chunk.to_dict()
        for chunk in evidence_chunks
    ]

    OUTPUT_PATH.write_text(
        json.dumps(
            evidence_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"PDF：{PDF_PATH.name}")
    print(f"处理页数：{len(cleaned_pages)}")
    print(f"证据数量：{len(evidence_chunks)}")
    print(f"输出位置：{OUTPUT_PATH}")

    print("\n前 3 条证据：")

    for chunk in evidence_chunks[:3]:
        print(
            f"\n{chunk.evidence_id} | "
            f"第 {chunk.page_number} 页 | "
            f"页内第 {chunk.chunk_index} 段"
        )
        print(chunk.text[:200])


if __name__ == "__main__":
    main()
