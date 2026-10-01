"""调用视觉模型描述第六天选定的三个PDF页面。"""

import json
from pathlib import Path

from research_kb.vision_service import VisionService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "data" / "images"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "visual_descriptions.json"

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
    """生成三个页面的视觉描述并保存为JSON。"""
    service = VisionService()
    records: list[dict[str, str | int]] = []

    print(f"视觉模型：{service.model_name}")

    for index, (source, page_number) in enumerate(
        CHART_PAGES,
        start=1,
    ):
        image_path = IMAGE_ROOT / Path(source).stem / f"page_{page_number:04d}.png"

        print(f"\n正在分析 {index}/{len(CHART_PAGES)}：{source}，PDF第{page_number}页")

        result = service.describe_page(
            source=source,
            page_number=page_number,
            image_path=image_path,
        )

        # 把图片的绝对路径转换为相对路径
        relative_image_path = result.image_path.relative_to(PROJECT_ROOT).as_posix()

        records.append(
            {
                "source": result.source,
                "page_number": result.page_number,
                "image_path": relative_image_path,
                "description": result.description,
            }
        )

        print(result.description)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n处理完成：{len(records)}个页面")
    print(f"结果文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
