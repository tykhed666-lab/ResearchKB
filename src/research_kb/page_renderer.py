"""将指定PDF物理页渲染成视觉模型可以读取的PNG图片。"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from research_kb.settings import IMAGE_ROOT

DEFAULT_DPI = 144
MIN_DPI = 72
MAX_DPI = 200


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """记录一个已经渲染完成的PDF页面。"""

    source: str
    page_number: int
    image_path: Path
    width: int
    height: int


def get_page_image_path(
    source: str,
    page_number: int,
) -> Path:
    """根据PDF文件名和物理页码生成页面图片路径。"""
    if not source.strip():
        raise ValueError("source不能为空")

    if page_number <= 0:
        raise ValueError("page_number必须大于0")
    # stem就是去掉文件名后缀的意思
    source_stem = Path(source).stem

    return IMAGE_ROOT / source_stem / f"page_{page_number:04d}.png"


def render_pdf_page(
    pdf_path: str | Path,
    page_number: int,
    dpi: int = DEFAULT_DPI,
) -> RenderedPage:
    """把指定PDF物理页渲染成PNG图片。

    Args:
        pdf_path: 原始PDF文件路径。
        page_number: 用户看到的物理页码，从1开始。
        dpi: 输出图片分辨率。

    Returns:
        包含来源、页码、图片路径和尺寸的渲染结果。

    Raises:
        FileNotFoundError: PDF文件不存在。
        ValueError: 文件类型、页码或DPI不正确。
    """
    path = Path(pdf_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"找不到PDF文件：{path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是PDF：{path.name}")

    if page_number <= 0:
        raise ValueError("page_number必须大于0")

    if not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(f"dpi必须在{MIN_DPI}到{MAX_DPI}之间")

    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"PDF需要密码：{path.name}")

        if page_number > document.page_count:
            raise ValueError(
                f"PDF只有{document.page_count}页，无法渲染第{page_number}页"
            )

        # 渲染和页面展示共用同一种图片路径规则。
        image_path = get_page_image_path(
            source=path.name,
            page_number=page_number,
        )

        image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # PyMuPDF内部页码从0开始，所以这里需要减1。
        page = document.load_page(page_number - 1)

        # PDF的基础分辨率为72 DPI。
        scale = dpi / 72
        matrix = pymupdf.Matrix(scale, scale)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )
        pixmap.save(str(image_path))

        return RenderedPage(
            source=path.name,
            page_number=page_number,
            image_path=image_path,
            width=pixmap.width,
            height=pixmap.height,
        )
