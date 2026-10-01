"""测试问答链路的自适应召回和引用校验。"""

from unittest.mock import Mock

from research_kb.qa import ModelAnswer, RAGQuestionAnswerer
from research_kb.retrieval import RetrievalResult


def make_result(index: int) -> RetrievalResult:
    """构造不依赖真实 Milvus 的检索结果。"""
    return RetrievalResult(
        evidence_id=f"evidence_{index}",
        score=1 - index / 100,
        source="demo.pdf",
        page_number=index,
        chunk_index=0,
        content_type="text",
        text=f"证据 {index}",
    )


def test_answer_widens_retrieval_after_insufficient_context() -> None:
    """前五条证据不足时，应扩大一次召回并使用新增证据回答。"""
    initial_results = [make_result(index) for index in range(1, 6)]
    widened_results = [make_result(index) for index in range(1, 8)]

    retriever = Mock()
    retriever.search.side_effect = [initial_results, widened_results]
    retriever.expand_top_result_page.return_value = initial_results

    answerer = RAGQuestionAnswerer.__new__(RAGQuestionAnswerer)
    answerer.retriever = retriever
    answerer.top_k = 5
    answerer.model_name = "test-model"
    answerer._generate_model_answer = Mock(
        side_effect=[
            ModelAnswer(
                answer="证据不足",
                insufficient_information=True,
                missing_information="缺少目标页",
            ),
            ModelAnswer(
                answer="已从扩大召回结果中找到答案",
                cited_evidence_ids=["evidence_7"],
                insufficient_information=False,
            ),
        ]
    )

    result = answerer.answer("测试问题")

    assert result.insufficient_information is False
    assert result.retrieved_count == 7
    assert [citation.evidence_id for citation in result.citations] == ["evidence_7"]
    assert retriever.search.call_count == 2
    assert retriever.search.call_args_list[1].kwargs["top_k"] == 10


def test_insufficient_answer_does_not_expose_irrelevant_citations() -> None:
    """拒答时不应把无关的召回结果展示成答案来源。"""
    retrieved_results = [make_result(1)]
    model_answer = ModelAnswer(
        answer="现有证据不足",
        cited_evidence_ids=["evidence_1"],
        insufficient_information=True,
        missing_information="缺少相关公司资料",
    )

    citations = RAGQuestionAnswerer._validate_citations(
        model_answer=model_answer,
        retrieved_results=retrieved_results,
    )

    assert citations == ()
