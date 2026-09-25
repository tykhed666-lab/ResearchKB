"""检查 PDF 加载和文本清理结果。"""

from pathlib import Path

from research_kb.pdf_loader import load_pdf_pages
from research_kb.text_cleaner import clean_pdf_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "01_NVDA_2026_Annual_Report.pdf"


def main() -> None:
    """加载并清理 NVIDIA 年报，展示前三页的处理结果。"""
    raw_pages = load_pdf_pages(PDF_PATH)
    cleaned_pages = clean_pdf_pages(raw_pages)

    empty_pages = [page for page in cleaned_pages if page.is_empty]

    print(f"文件：{PDF_PATH.name}")
    print(f"总页数：{len(cleaned_pages)}")
    print(f"空文本页数：{len(empty_pages)}")

    for raw_page, cleaned_page in zip(
        raw_pages[:3],
        cleaned_pages[:3],
        strict=True,
    ):
        preview = cleaned_page.text[:200] or "[空文本页]"

        print(f"\n第 {cleaned_page.page_number} 页")
        print(f"原始字符数：{raw_page.char_count}")
        print(f"清理后字符数：{cleaned_page.char_count}")
        print(f"清理后预览：{preview}")


if __name__ == "__main__":
    main()