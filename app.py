"""ResearchKB 的 Streamlit 演示页面。"""

from pathlib import Path

import streamlit as st

from research_kb.document_registry import (
    DocumentMetadata,
    DocumentRegistry,
    STATUS_INDEXED,
)
from research_kb.embedding_service import EmbeddingService
from research_kb.indexer import EvidenceIndexer
from research_kb.milvus_store import MilvusStore
from research_kb.page_renderer import (
    get_page_image_path,
)
from research_kb.qa import QAResult, RAGQuestionAnswerer
from research_kb.retrieval import MilvusRetriever
from research_kb.settings import DEFAULT_TOP_K
from research_kb.upload_service import (
    PdfUploadService,
    UploadReport,
)

# streamlit的页面全局配置
st.set_page_config(
    page_title="ResearchKB",
    page_icon="📚",
    layout="wide",
)


# 创建并缓存数据库、Embedding和问答服务
@st.cache_resource(show_spinner=False)
def create_services() -> tuple[
    MilvusStore,
    RAGQuestionAnswerer,
    PdfUploadService,
    DocumentRegistry,
]:
    """创建并缓存问答、上传和文档注册服务。"""
    store = MilvusStore()

    embedding_service = (
        EmbeddingService()
    )

    retriever = MilvusRetriever(
        store=store,
        embedding_service=embedding_service,
    )

    answerer = RAGQuestionAnswerer(
        retriever=retriever,
        top_k=DEFAULT_TOP_K,
    )

    indexer = EvidenceIndexer(
        store=store,
        embedding_service=embedding_service,
    )

    # SQLite 注册表负责管理一份文档的
    # 元数据、处理状态和证据数量。
    registry = DocumentRegistry()

    upload_service = PdfUploadService(
        indexer=indexer,
        registry=registry,
    )

    return (
        store,
        answerer,
        upload_service,
        registry,
    )


def render_qa_result(result: QAResult) -> None:
    """展示回答、信息缺口和引用证据。"""
    if result.insufficient_information:
        st.warning("当前知识库证据不足")
    else:
        st.success("已依据知识库证据生成回答")

    st.markdown(result.answer)

    if result.missing_information:
        st.caption(
            f"缺少的信息：{result.missing_information}"
        )

    st.markdown("#### 引用证据")

    if not result.citations:
        st.info("本次回答没有可展示的引用证据。")
        return

    for index, citation in enumerate(
        result.citations,
        start=1,
    ):
        evidence_type = (
            "图表证据"
            if citation.content_type == "image"
            else "文本证据"
        )

        label = (
            f"{index}. {citation.source} · "
            f"PDF第{citation.page_number}页 · "
            f"{evidence_type} · "
            f"相似度 {citation.score:.4f}"
        )

        with st.expander(label):
            st.code(
                citation.evidence_id,
                language=None,
            )

            if citation.content_type == "image":
                image_path = get_page_image_path(
                    source=citation.source,
                    page_number=citation.page_number,
                )

                if image_path.is_file():
                    st.image(
                        str(image_path),
                        caption=(
                            f"{citation.source} "
                            f"PDF第{citation.page_number}页"
                        ),
                    )
                else:
                    st.warning(
                        "本地原图不存在，但视觉描述仍可正常使用。"
                    )

                st.markdown("**视觉模型描述**")

            st.write(citation.text)


def render_message(message: dict) -> None:
    """重新展示一条历史聊天消息。"""

    with st.chat_message(message["role"]):
        # 渲染用户消息
        if message["role"] == "user":
            st.markdown(message["content"])

            if message.get("source"):
                st.caption(
                    f"限定资料：{message['source']}"
                )
            return

        if message.get("error"):
            st.error(message["error"])
            return
        # 渲染助手消息
        render_qa_result(message["result"])


def render_upload_report(
    report: UploadReport,
) -> None:
    """在侧边栏展示PDF上传和入库结果。"""
    st.success("PDF处理完成")

    if report.duplicate:
        st.info(
            "相同内容已经成功入库，"
            "本次没有重复生成向量。"
        )
    st.caption(f"保存文件：{report.saved_name}")

    first_column, second_column = st.columns(2)

    with first_column:
        st.metric(
            "处理页数",
            report.processed_pages,
        )
        st.metric(
            "新增证据",
            report.inserted_chunks,
        )

    with second_column:
        st.metric(
            "证据总数",
            report.evidence_chunks,
        )
        st.metric(
            "跳过证据",
            report.skipped_chunks,
        )

    if report.empty_pages:
        st.warning(
            f"发现{report.empty_pages}个空文本页，"
            f"当前版本未对其执行OCR。"
        )

    if report.truncated:
        st.warning(
            f"PDF共有{report.total_pages}页，"
            f"本次只处理前{report.processed_pages}页。"
        )

    if report.renamed:
        st.info(
            "发现同名但内容不同的文件，"
            "已使用文件哈希自动重命名。"
        )


def main() -> None:
    """渲染ResearchKB页面并处理用户问题。"""
    st.title("📚 ResearchKB")
    st.caption(
        "个人多模态行研资料库：依据已上传文档回答，"
        "并提供原文页码。"
    )

    try:
        (
            store,
            answerer,
            upload_service,
            registry,
        ) = create_services()

        entity_count = (
            store.get_entity_count()
        )

        # 文档目录改为从 SQLite 读取，
        # 不再扫描全部 Milvus 向量推断文件列表。
        documents = (
            registry.list_documents()
        )

        # 只有成功入库的文档才能参与检索。
        sources = [
            document.saved_name
            for document in documents
            if (
                document.status
                == STATUS_INDEXED
            )
        ]

        service_error = None

    except Exception as error:
        store = None
        answerer = None
        upload_service = None
        registry = None
        entity_count = 0
        documents = []
        sources = []
        service_error = str(error)

    # st.sidebar 以下内容全被渲染到左侧边框
    with st.sidebar:
        # 显示标题
        st.header("知识库状态")
        # 显示一个数字指标卡片
        st.metric(
            label="Milvus实体数",
            value=entity_count,
        )
        st.metric(
            label="已登记文档",
            value=len(documents),
        )

        if service_error:
            st.error(
                "知识库服务连接失败，请检查Milvus和模型配置。"
            )

        # 展示已查重的全部资料
        source_options = [
            "全部资料",
            *sources,
        ]

        # 创建一个下拉选择框，选择检索范围
        selected_source = st.selectbox(
            label="检索范围",
            options=source_options,
        )

        st.markdown(
            "#### 文档登记簿"
        )

        if not documents:
            st.info(
                "当前没有已登记资料。"
            )

        else:
            for document in documents:
                label = document.title

                if document.ticker:
                    label += (
                        f" ({document.ticker})"
                    )

                st.markdown(
                    f"**{label}**"
                )

                st.caption(
                    f"{document.document_type}"
                    f" · "
                    f"{document.industry or '行业未填'}"
                    f" · 状态 {document.status}"
                    f" · 文本 {document.text_chunk_count}"
                    f" · 图像 {document.image_chunk_count}"
                )

                if document.error_message:
                    st.error(
                        document.error_message
                    )

        # PDF上传区域从这里开始。
        st.divider()
        st.subheader("上传PDF")

        last_upload_report = (
            st.session_state.pop(
                "last_upload_report",
                None,
            )
        )

        if last_upload_report is not None:
            render_upload_report(
                last_upload_report
            )

        uploaded_file = st.file_uploader(
            label="选择PDF文件",
            type=["pdf"],
            accept_multiple_files=False,
            help="单个文件不超过20MB",
        )

        # file_uploader 选中的文件改变时，自动用文件名初始化标题。
        # 之后用户仍可在输入框中修改标题。
        current_upload_name = (
            uploaded_file.name
            if uploaded_file is not None
            else None
        )

        if (
            st.session_state.get(
                "upload_metadata_file_name"
            )
            != current_upload_name
        ):
            st.session_state[
                "upload_metadata_file_name"
            ] = current_upload_name
            st.session_state[
                "upload_document_title"
            ] = (
                Path(current_upload_name).stem
                if current_upload_name
                else ""
            )

        document_title = st.text_input(
            label="文档标题",
            key="upload_document_title",
            help="用于文档目录展示，可与原始文件名不同。",
        )

        company = st.text_input(
            label="公司或机构",
            placeholder="例如：NVIDIA",
        )

        ticker = st.text_input(
            label="股票代码",
            placeholder="例如：NVDA，可不填",
        )

        industry = st.text_input(
            label="行业",
            placeholder="例如：半导体、消费零售",
        )

        document_type = st.selectbox(
            label="文档类型",
            options=[
                "年报",
                "季报",
                "投资者演示",
                "行业报告",
                "研报",
                "其他",
            ],
        )

        report_date = st.text_input(
            label="报告日期",
            placeholder="YYYY-MM-DD，可不填",
            help="只填写能够从资料中确认的日期。",
        )

        max_pages = st.number_input(
            label="本次最多处理页数",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help=(
                "演示版本限制为30页，"
                "避免意外产生大量Embedding费用。"
            ),
        )

        upload_disabled = (
            uploaded_file is None
            or upload_service is None
            or not document_title.strip()
        )

        if st.button(
            "解析并入库",
            disabled=upload_disabled,
            use_container_width=True,
        ):
            with st.spinner(
                "正在解析、生成向量并写入Milvus……"
            ):
                try:
                    upload_report = (
                        upload_service.ingest_pdf(
                            file_name=uploaded_file.name,
                            file_bytes=(
                                uploaded_file.getvalue()
                            ),
                            metadata=DocumentMetadata(
                                title=document_title,
                                company=company,
                                ticker=ticker,
                                industry=industry,
                                document_type=document_type,
                                report_date=report_date,
                            ),
                            max_pages=int(max_pages),
                        )
                    )

                    st.session_state[
                        "last_upload_report"
                    ] = upload_report

                    # 重新运行页面，刷新实体数和文档列表。
                    st.rerun()

                except ValueError as error:
                    st.error(str(error))

                except Exception as error:
                    st.error(
                        "PDF入库失败，请检查模型、"
                        "Milvus和网络状态。"
                    )
                    print(
                        f"Streamlit上传错误："
                        f"{type(error).__name__}: {error}"
                    )

        # PDF上传区域到这里结束。
        st.divider()

        # 清空对话按钮继续放在上传区域下面。
        if st.button(
            "清空对话",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

    # 初始化聊天记录，只在第一次执行
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 遍历所有聊天消息
    for message in st.session_state.messages:
        render_message(message)

    knowledge_base_ready = (
        answerer is not None
        and entity_count > 0
    )

    # 用户输入的文本
    prompt = st.chat_input(
        placeholder=(
            "请根据已上传的行业资料提问"
            if knowledge_base_ready
            else "知识库当前不可用"
        ),
        disabled=not knowledge_base_ready,
        # 提交后暂时禁用输入框，避免重复点击。
        submit_mode="disable",
    )

    if prompt is None:
        return
    # 将用户输入文本转换为问题文本
    question = prompt.strip()

    if not question:
        st.warning("问题不能为空。")
        return

    # 这是加了一个限定，如果要搜索全部资料，不加filter
    # 不然就要限定来源，即只搜索选中的来源
    # 限定来源，即只搜索选中的来源
    source_filter = (
        None
        if selected_source == "全部资料"
        else selected_source
    )

    user_message = {
        "role": "user",
        "content": question,
        "source": source_filter,
    }

    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("正在检索证据并生成回答……"):
            try:
                result = answerer.answer(
                    question=question,
                    source=source_filter,
                )

                assistant_message = {
                    "role": "assistant",
                    "result": result,
                }

                render_qa_result(result)

            except Exception as error:
                # 页面只显示简洁错误，避免泄露API配置。
                error_message = (
                    "问答执行失败，请检查模型、网络和Milvus状态。"
                )

                assistant_message = {
                    "role": "assistant",
                    "error": error_message,
                }

                st.error(error_message)

                # 详细错误只写入服务器终端，方便开发时排查。
                print(
                    f"Streamlit问答错误："
                    f"{type(error).__name__}: {error}"
                )

    st.session_state.messages.append(
        assistant_message
    )


if __name__ == "__main__":
    main()
