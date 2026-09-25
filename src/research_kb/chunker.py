"""把清理后的 PDF 页面切分成可检索的证据片段。"""

from dataclasses import asdict, dataclass
from hashlib import sha256

from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_kb.pdf_loader import PdfPage


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    """表示一个可以存入向量数据库的证据片段。

    Attributes:
        evidence_id: 证据片段的唯一标识。
        source: 原始 PDF 文件名。
        page_number: 证据所在页码。
        chunk_index: 当前证据在页面中的序号。
        content_type: 证据类型，目前只处理文本。
        text: 证据正文。
    """

    evidence_id: str
    source: str
    page_number: int
    chunk_index: int
    content_type: str
    text: str

    def to_dict(self) -> dict[str, str | int]:
        """转换成可以写入 JSON 的字典。"""
        return asdict(self)


def build_evidence_id(
    source: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """根据证据来源和内容生成稳定的证据 ID。

    相同的文件、页码、序号和文本会生成相同 ID；
    文本发生变化时，ID 也会变化。
    """
    raw_value = f"{source}|{page_number}|{chunk_index}|{text}"
    digest = sha256(raw_value.encode("utf-8")).hexdigest()[:16]

    return f"text_{digest}"


def split_pages_into_chunks(
    pages: list[PdfPage],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[EvidenceChunk]:
    """把多个 PDF 页面切分成证据片段。

    Args:
        pages: 已经完成文本清理的页面。
        chunk_size: 每个片段允许的最大字符数。
        chunk_overlap: 相邻片段重复保留的字符数。

    Returns:
        带有来源信息和证据 ID 的片段列表。

    Raises:
        ValueError: 切分参数不合理。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0，并且小于 chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 优先在段落、句子和单词边界切分，最后才按字符切分。
        separators=["\n\n", "\n", ". ", "。", "！", "？", " ", ""],
    )

    evidence_chunks: list[EvidenceChunk] = []

    for page in pages:
        # 空页面没有可检索内容，不生成证据。
        if page.is_empty:
            continue

        page_chunks = splitter.split_text(page.text)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            evidence_chunks.append(
                EvidenceChunk(
                    evidence_id=build_evidence_id(
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    ),
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    content_type="text",
                    text=chunk_text,
                )
            )

    return evidence_chunks
