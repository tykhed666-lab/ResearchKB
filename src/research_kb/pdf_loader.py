"""负责读取 PDF，并把每一页转换成结构化的 Python 对象。"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True, slots=True)
class PdfPage:
    """表示从 PDF 中解析出来的一页内容。

    Attributes:
        source: PDF 文件名。
        page_number: 面向用户的页码，从 1 开始。
        text: 当前页提取出的文本。
    """

    source: str
    page_number: int
    text: str

    @property
    def char_count(self) -> int:
        """返回当前页的字符数量。"""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """判断当前页是否没有提取到文字。"""
        return not self.text.strip()


def load_pdf_pages(pdf_path: str | Path) -> list[PdfPage]:
    """按页读取一个 PDF 文件。

    Args:
        pdf_path: PDF 文件路径，可以传入字符串或 Path 对象。

    Returns:
        按原始顺序排列的 PDF 页面列表。

    Raises:
        FileNotFoundError: PDF 文件不存在。
        ValueError: 输入文件不是 PDF，或者 PDF 需要密码。
    """
    path = Path(pdf_path).resolve()
    # .resolve() 把相对路径转成绝对路径，避免后续出现路径歧义。

    if not path.is_file():
        raise FileNotFoundError(f"找不到 PDF 文件：{path}")
    # 先检查文件是否存在，再检查扩展名是否为 .pdf。
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"输入文件不是 PDF：{path.name}")

    pages: list[PdfPage] = []

    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError(f"PDF 需要密码，暂时无法解析：{path.name}")

        for page_index, page in enumerate(document):
            # sort=True 会尝试按照页面上的阅读顺序排列文字。
            text = page.get_text("text", sort=True).strip()

            pages.append(
                PdfPage(
                    source=path.name,
                    # PyMuPDF 从 0 计数，但用户看到的 PDF 页码从 1 开始。
                    page_number=page_index + 1,
                    text=text,
                )
            )

    return pages
