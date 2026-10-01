"""清理从 PDF 页面中提取出的文本。"""

import re
from dataclasses import replace

from research_kb.pdf_loader import PdfPage

# 匹配“3”“Page 3”“3 / 175”“3 of 175”等独立页码。
# 用正则表达式来匹配pdf的页码行
# flags=re.IGNORECASE 忽略大小写
PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?$",
    flags=re.IGNORECASE,
)


def clean_page_text(text: str) -> str:
    """清理单页 PDF 文本中的常见噪声。

    Args:
        text: PDF 页面原始文本。

    Returns:
        清理后的单行文本。
    """
    # PDF 中的不间断空格看起来像普通空格，但处理方式不同。
    # \u00a0 是 不间断空格，为了方便，把它直接替换为空格
    text = text.replace("\u00a0", " ")

    cleaned_lines: list[str] = []
    # splitlines()：按换行符拆分成一行一行
    for raw_line in text.splitlines():
        # 合并一行中连续的空格和制表符。
        # [\t] 把连续空格匹配成一个空格
        line = re.sub(r"[ \t]+", " ", raw_line).strip()

        if not line:
            continue

        # 删除独立存在的页码，但不会删除正文中的普通数字。
        if PAGE_NUMBER_PATTERN.fullmatch(line):
            continue

        cleaned_lines.append(line)

    # 使用空格连接各行，减少 PDF 强制换行造成的干扰。
    cleaned_text = " ".join(cleaned_lines)

    # 修复单词因为换行而产生的断词，例如 “inter- national”。
    cleaned_text = re.sub(r"(?<=\w)-\s+(?=\w)", "", cleaned_text)

    # 再次合并剩余的连续空白。
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def clean_pdf_pages(pages: list[PdfPage]) -> list[PdfPage]:
    """清理多个 PDF 页面，同时保留原文件名和页码。

    Args:
        pages: PDF 加载模块返回的页面列表。

    Returns:
        文本已经清理过的新页面列表。
    """
    cleaned_pages: list[PdfPage] = []

    for page in pages:
        # PdfPage 是不可修改的数据类，因此创建一个新对象替换 text。
        cleaned_page = replace(
            page,
            text=clean_page_text(page.text),
        )
        cleaned_pages.append(cleaned_page)

    return cleaned_pages
