"""把 SEC 主 HTML 转换成适合向量检索的正文证据。"""

from html.parser import HTMLParser
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_kb.chunker import (
    EvidenceChunk,
    build_evidence_id,
)
from research_kb.external_source_registry import (
    ExternalSourceRecord,
)
from research_kb.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)


BLOCK_TAGS = frozenset(
    {
        "address", "article", "br", "caption", "div",
        "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "li", "main", "p", "section", "table",
        "td", "th", "tr", "ul", "ol",
    }
)

IGNORED_TAGS = frozenset(
    {
        "script", "style", "noscript", "template",
        # Inline XBRL 的隐藏区含大量重复机器数据，
        # 用户在网页中看不到，不应重复进入检索。
        "ix:hidden",
    }
)


class _VisibleTextParser(HTMLParser):
    """只收集网页中用户可见的文本。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    @staticmethod
    def _is_hidden(
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> bool:
        attributes = {
            name.lower(): (value or "").lower()
            for name, value in attrs
        }
        style = attributes.get("style", "").replace(" ", "")
        return (
            tag in IGNORED_TAGS
            or "hidden" in attributes
            or attributes.get("aria-hidden") == "true"
            or "display:none" in style
            or "visibility:hidden" in style
        )

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if self._ignored_depth > 0:
            self._ignored_depth += 1
            return
        if self._is_hidden(tag, attrs):
            self._ignored_depth = 1
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if (
            self._ignored_depth == 0
            and not self._is_hidden(tag.lower(), attrs)
            and tag.lower() in BLOCK_TAGS
        ):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if tag.lower() in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)


def extract_visible_sec_text(
    html_content: bytes | str,
) -> str:
    """提取 SEC HTML 可见正文并保留段落边界。"""
    if isinstance(html_content, bytes):
        html_text = html_content.decode(
            "utf-8",
            errors="replace",
        )
    else:
        html_text = html_content

    parser = _VisibleTextParser()
    parser.feed(html_text)
    parser.close()

    text = "".join(parser.parts).replace("\xa0", " ")
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)

    return "\n".join(cleaned_lines)


def split_sec_text_into_chunks(
    record: ExternalSourceRecord,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chunks: int = 500,
) -> tuple[list[EvidenceChunk], bool]:
    """将 SEC 正文切成页码为 0 的外部文本证据。

    返回值中的布尔量表示是否因为成本上限截断。每条证据都带
    公司、表单和报告期前缀，避免模型混淆不同年份的指标。
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("SEC HTML 没有可用正文")
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap 必须大于等于 0，并且小于 chunk_size"
        )
    if max_chunks <= 0:
        raise ValueError("max_chunks 必须大于 0")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", "；", " ", ""],
    )
    raw_chunks = splitter.split_text(cleaned_text)
    truncated = len(raw_chunks) > max_chunks
    raw_chunks = raw_chunks[:max_chunks]

    period = record.report_date or record.filing_date
    prefix = (
        f"[SEC {record.form_type} | {record.company_name} "
        f"({record.ticker}) | 报告期 {period}] "
    )

    chunks: list[EvidenceChunk] = []
    for chunk_index, raw_text in enumerate(raw_chunks, start=1):
        chunk_text = f"{prefix}{raw_text.strip()}"
        if not raw_text.strip():
            continue
        chunks.append(
            EvidenceChunk(
                evidence_id=build_evidence_id(
                    source=record.source_id,
                    page_number=0,
                    chunk_index=chunk_index,
                    text=chunk_text,
                ),
                source=record.source_id,
                page_number=0,
                chunk_index=chunk_index,
                content_type="sec_html",
                text=chunk_text,
            )
        )

    return chunks, truncated
