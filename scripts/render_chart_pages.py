"""渲染第六天首轮使用的三个多模态页面。"""

from pathlib import Path

from research_kb.page_renderer import render_pdf_page


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

CHART_PAGES = (
    (
        "01_NVDA_2026_Annual_Report.pdf",
        3,
    ),
    (
        "03_LITE_OFC_2026_Investor_Briefing.pdf",
        30,
    ),
    (
        "05_STX_2025_Decarbonizing_Data_Report.pdf",
        11,
    ),
)


def main() -> None:
    """渲染并输出三个页面的图片信息。"""
    for file_name, page_number in CHART_PAGES:
        result = render_pdf_page(
            pdf_path=RAW_DATA_DIR / file_name,
            page_number=page_number,
            dpi=144,
        )

        print(
            f"{result.source} | "
            f"PDF第{result.page_number}页"
        )
        print(
            f"图片尺寸："
            f"{result.width}x{result.height}"
        )
        print(f"保存位置：{result.image_path}")
        print()


if __name__ == "__main__":
    main()