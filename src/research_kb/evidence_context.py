"""把检索结果转换成模型可以阅读的统一证据上下文。

普通问答和研究简报都需要向模型提供证据。把格式化规则集中在这里，
可以避免两条业务流程对 SEC、PDF 页码和来源日期产生不同解释。
"""

from collections.abc import Sequence

from research_kb.retrieval import RetrievalResult


def format_evidence_location(result: RetrievalResult) -> str:
    """返回一条证据的位置描述，明确区分网页和 PDF。"""
    if result.source_type == "sec":
        return f"SEC官方原文：{result.source_url or '链接未提供'}"
    return f"PDF物理页码：{result.page_number}"


def build_evidence_context(
    results: Sequence[RetrievalResult],
    *,
    include_score: bool,
) -> str:
    """构建发送给模型的证据文本。

    Args:
        results: 按检索顺序排列的证据。
        include_score: 是否向模型展示向量相似度。普通问答保留分数，研究
            简报只需要来源和正文，因此可以关闭。
    """
    blocks: list[str] = []

    for result in results:
        lines = [
            f"证据ID：{result.evidence_id}",
            f"证据类型：{result.content_type}",
            f"来源名称：{result.display_title or result.source}",
            f"来源类型：{result.source_type}",
            f"来源日期：{result.source_date or '未提供'}",
            format_evidence_location(result),
        ]
        if include_score:
            lines.append(f"相似度：{result.score:.4f}")
        lines.append(f"正文：{result.text}")
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)
