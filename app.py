"""ResearchKB的Streamlit演示页面。"""

import streamlit as st
from research_kb.indexer import EvidenceIndexer
from research_kb.upload_service import (
    PdfUploadService,
    UploadReport,
)
from research_kb.embedding_service import EmbeddingService
from research_kb.milvus_store import MilvusStore
from research_kb.qa import QAResult, RAGQuestionAnswerer
from research_kb.retrieval import MilvusRetriever

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
]:
    """创建并缓存问答与上传服务。"""
    store = MilvusStore()
    embedding_service = EmbeddingService()

    retriever = MilvusRetriever(
        store=store,
        embedding_service=embedding_service,
    )

    answerer = RAGQuestionAnswerer(
        retriever=retriever,
        top_k=5,
    )

    indexer = EvidenceIndexer(
        store=store,
        embedding_service=embedding_service,
    )

    upload_service = PdfUploadService(
        indexer=indexer,
    )

    return store, answerer, upload_service


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
        label = (
            f"{index}. {citation.source} · "
            f"PDF第{citation.page_number}页 · "
            f"相似度 {citation.score:.4f}"
        )

        with st.expander(label):
            st.code(
                citation.evidence_id,
                language=None,
            )
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
        "面向个人的AI基础设施多模态RAG行研知识库"
    )

    try:
        store, answerer, upload_service = create_services()
        entity_count = store.get_entity_count()
        sources = store.list_sources()
        service_error = None
    except Exception as error:
        store = None
        answerer = None
        upload_service = None
        entity_count = 0
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
            label="已入库文档",
            value=len(sources),
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

        st.markdown("#### 已入库资料")

        if not sources:
            st.info("当前没有已入库资料。")
        else:
            # 遍历所有入库资料
            for source in sources:
                st.caption(f"• {source}")

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

    prompt = st.chat_input(
        placeholder=(
            "请输入AI基础设施行业问题"
            if knowledge_base_ready
            else "知识库当前不可用"
        ),
        disabled=not knowledge_base_ready,
        # 提交后暂时禁用输入框，避免重复点击。
        submit_mode="disable",
    )

    if prompt is None:
        return

    question = prompt.strip()

    if not question:
        st.warning("问题不能为空。")
        return

    # 这是加了一个限定，如果要搜索全部资料，不加filter
    # 不然就要限定来源，即只搜索选中的来源
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