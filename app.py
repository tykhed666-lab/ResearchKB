"""ResearchKB 的 Streamlit 演示页面。"""

from datetime import date
from pathlib import Path

import streamlit as st

from research_kb.document_service import (
    DocumentManagementService,
)
from research_kb.document_registry import (
    DocumentMetadata,
    DocumentRegistry,
    STATUS_INDEXED,
)
from research_kb.embedding_service import EmbeddingService
from research_kb.external_ingestion import (
    ExternalIngestionService,
)
from research_kb.external_source_registry import (
    EXTERNAL_STATUS_INDEXED,
    ExternalSourceRegistry,
)
from research_kb.indexer import EvidenceIndexer
from research_kb.milvus_store import MilvusStore
from research_kb.page_renderer import (
    get_page_image_path,
)
from research_kb.qa import (
    CitationValidationError,
    QAResult,
    RAGQuestionAnswerer,
)
from research_kb.research_report import ResearchReportService
from research_kb.retrieval import (
    MilvusRetriever,
    RetrievalSourceMetadata,
)
from research_kb.sec_edgar import (
    SecEdgarClient,
    SecEdgarError,
)
from research_kb.settings import DEFAULT_TOP_K
from research_kb.upload_service import (
    PdfUploadService,
    UploadReport,
)
from research_kb.vision_service import (
    VisionService,
)
from research_kb.visual_ingestion import (
    VisualIngestionReport,
    VisualIngestionService,
    parse_page_numbers,
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
    DocumentManagementService,
    VisualIngestionService,
    EvidenceIndexer,
    ResearchReportService,
]:
    """创建并缓存问答、上传和文档管理服务。"""
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

    research_report_service = ResearchReportService(
        retriever=retriever,
        top_k=12,
    )

    indexer = EvidenceIndexer(
        store=store,
        embedding_service=embedding_service,
    )

    # SQLite注册表负责管理文档元数据、
    # 项目归属、状态和证据数量。
    registry = DocumentRegistry()

    upload_service = PdfUploadService(
        indexer=indexer,
        registry=registry,
    )

    document_manager = (
        DocumentManagementService(
            store=store,
            registry=registry,
            upload_service=upload_service,
        )
    )

    # VisionService只负责单张图片的模型调用；
    # VisualIngestionService负责完整的多页业务流程。
    vision_service = VisionService()

    visual_ingestion_service = (
        VisualIngestionService(
            store=store,
            indexer=indexer,
            vision_service=(
                vision_service
            ),
            registry=registry,
        )
    )

    return (
        store,
        answerer,
        upload_service,
        registry,
        document_manager,
        visual_ingestion_service,
        indexer,
        research_report_service,
    )


@st.cache_resource(show_spinner=False)
def create_sec_services() -> tuple[
    SecEdgarClient,
    ExternalSourceRegistry,
]:
    """创建并缓存 SEC 客户端和外部资料登记簿。

    SEC 服务与 Milvus 问答服务分开初始化，
    因此 SEC 配置错误不会导致本地知识库不可用。
    """
    return (
        SecEdgarClient(),
        ExternalSourceRegistry(),
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
        if citation.source_type == "sec":
            evidence_type = "SEC网页证据"
            location = "SEC官方原文"
        else:
            evidence_type = (
                "图表证据"
                if citation.content_type == "image"
                else "文本证据"
            )
            location = f"PDF第{citation.page_number}页"

        title = citation.display_title or citation.source
        label = (
            f"{index}. {title} · {location} · "
            f"{evidence_type} · 相似度 {citation.score:.4f}"
        )

        with st.expander(label):
            st.code(
                citation.evidence_id,
                language=None,
            )

            if (
                citation.source_type == "sec"
                and citation.source_url
            ):
                st.link_button(
                    "打开SEC官方原文",
                    citation.source_url,
                    use_container_width=True,
                )
                if citation.source_date:
                    st.caption(
                        f"报告期或提交日期：{citation.source_date}"
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
                source_scope = (
                    message["source"]
                )

                if isinstance(
                    source_scope,
                    list,
                ):
                    source_text = "、".join(
                        source_scope
                    )
                else:
                    source_text = (
                        source_scope
                    )

                st.caption(
                    f"限定资料：{source_text}"
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

def render_visual_ingestion_report(
    report: VisualIngestionReport,
) -> None:
    """展示多页视觉处理的汇总和逐页结果。"""
    if report.failed_pages:
        failed_text = "、".join(
            str(page_number)
            for page_number
            in report.failed_pages
        )

        st.warning(
            "图表页处理完成，但以下页面失败："
            f"{failed_text}"
        )

    else:
        st.success(
            "所选图表页处理完成"
        )

    st.caption(
        f"来源文件：{report.source}"
    )

    first_column, second_column = (
        st.columns(2)
    )

    with first_column:
        st.metric(
            label="视觉模型调用",
            value=report.model_calls,
        )
        st.metric(
            label="新增图像证据",
            value=(
                report.inserted_chunks
            ),
        )

    with second_column:
        st.metric(
            label="跳过已有证据",
            value=(
                report.skipped_chunks
            ),
        )
        st.metric(
            label="失败页面",
            value=len(
                report.failed_pages
            ),
        )

    st.markdown(
        "##### 逐页结果"
    )

    status_names = {
        "inserted": "新增",
        "skipped": "已存在",
        "failed": "失败",
    }

    for page_result in (
        report.page_results
    ):
        status_name = status_names.get(
            page_result.status,
            page_result.status,
        )

        label = (
            f"PDF第{page_result.page_number}页"
            f" · {status_name}"
        )

        with st.expander(label):
            st.code(
                page_result.evidence_id,
                language=None,
            )

            if (
                page_result.image_path
                is not None
                and page_result
                .image_path.is_file()
            ):
                st.image(
                    str(
                        page_result
                        .image_path
                    ),
                    caption=(
                        f"{report.source} "
                        f"PDF第"
                        f"{page_result.page_number}页"
                    ),
                )

            if page_result.error_message:
                if (
                    page_result.status
                    == "failed"
                ):
                    st.error(
                        page_result
                        .error_message
                    )
                else:
                    # 证据已经存在但原图恢复失败时，
                    # 检索仍可用，因此展示警告。
                    st.warning(
                        page_result
                        .error_message
                    )



def main() -> None:
    """渲染ResearchKB页面并处理用户问题。"""
    st.title("📚 ResearchKB")
    st.caption(
        "个人多模态行研资料库：依据PDF与SEC官方披露回答，"
        "并提供原文页码或官方链接。"
    )

    try:
        (
            store,
            answerer,
            upload_service,
            registry,
            document_manager,
            visual_ingestion_service,
            indexer,
            research_report_service,
        ) = create_services()

        entity_count = (
            store.get_entity_count()
        )

        # 页面启动时读取全部文档和研究项目。
        # 后续筛选仍会通过 Registry 查询 SQLite。
        documents = (
            registry.list_documents()
        )

        projects = (
            registry.list_projects()
        )

        service_error = None

    except Exception as error:
        store = None
        answerer = None
        document_manager = None
        upload_service = None
        registry = None
        entity_count = 0
        documents = []
        projects = []
        service_error = str(error)
        visual_ingestion_service = None
        indexer = None
        research_report_service = None

    # SEC 服务独立初始化。即使 SEC User-Agent
    # 配置不正确，本地 PDF 问答仍然可以使用。
    try:
        (
            sec_client,
            external_source_registry,
        ) = create_sec_services()

        external_source_count = (
            external_source_registry
            .count_sources()
        )
        sec_service_error = None

    except Exception as error:
        sec_client = None
        external_source_registry = None
        external_source_count = 0
        sec_service_error = str(error)

    external_ingestion_service = None
    if (
        sec_client is not None
        and external_source_registry is not None
        and indexer is not None
    ):
        external_ingestion_service = (
            ExternalIngestionService(
                client=sec_client,
                registry=external_source_registry,
                indexer=indexer,
            )
        )

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
        st.metric(
            label="已保存SEC披露",
            value=external_source_count,
        )

        if service_error:
            st.error(
                "知识库服务连接失败，请检查Milvus和模型配置。"
            )

        st.subheader("研究范围")
        # 显示项目创建、文档分配等操作完成后的提示。
        project_action_notice = (
            st.session_state.pop(
                "project_action_notice",
                None,
            )
        )

        if project_action_notice:
            st.success(project_action_notice)

        # 删除或重新入库完成后，
        # 在页面重新运行时显示结果。
        document_action_notice = (
            st.session_state.pop(
                "document_action_notice",
                None,
            )
        )

        if document_action_notice:
            st.success(
                document_action_notice
            )

        document_action_warnings = (
            st.session_state.pop(
                "document_action_warnings",
                (),
            )
        )

        for warning in (
            document_action_warnings
        ):
            st.warning(warning)

        # 删除文档后，在下一次页面运行、
        # 各个控件创建前清理旧的选择状态。
        if st.session_state.pop(
            "reset_document_widgets",
            False,
        ):
            for state_key in (
                "selected_document_sources",
                "document_scope_signature",
                "managed_document_id",
                "target_project_id",
                "managed_document_assignment_signature",
            ):
                st.session_state.pop(
                    state_key,
                    None,
                )

        # form 会把输入框和提交按钮组成一次完整操作，
        # 用户点击提交后才会真正调用 SQLite。
        with st.expander("新建研究项目"):
            with st.form(
                "create_project_form",
                clear_on_submit=True,
            ):
                new_project_name = (
                    st.text_input(
                        label="项目名称",
                        placeholder=(
                            "例如：新能源汽车"
                        ),
                    )
                )

                new_project_description = (
                    st.text_area(
                        label="项目说明",
                        placeholder=(
                            "记录该项目主要研究的"
                            "行业、公司或问题。"
                        ),
                    )
                )

                create_project_submitted = (
                    st.form_submit_button(
                        "创建项目",
                        use_container_width=True,
                    )
                )

            if create_project_submitted:
                if registry is None:
                    st.error(
                        "文档注册服务尚未连接。"
                    )

                else:
                    try:
                        created_project = (
                            registry.create_project(
                                name=(
                                    new_project_name
                                ),
                                description=(
                                    new_project_description
                                ),
                            )
                        )

                        # 把提示保存到 session_state，
                        # 页面重新运行后仍然能够显示。
                        st.session_state[
                            "project_action_notice"
                        ] = (
                            "已创建研究项目："
                            f"{created_project.name}"
                        )

                        # 重新读取项目列表，
                        # 让新项目立即出现在下拉框中。
                        st.rerun()

                    except ValueError as error:
                        st.error(str(error))

        project_by_id = {
            project.project_id: project
            for project in projects
        }

        project_options = [
            None,
            *project_by_id,
        ]

        selected_project_id = (
            st.selectbox(
                label="研究项目",
                options=project_options,
                format_func=lambda value: (
                    "全部项目"
                    if value is None
                    else project_by_id[
                        value
                    ].name
                ),
            )
        )

        industries = sorted(
            {
                document.industry
                for document in documents
                if document.industry
            }
        )

        selected_industry = (
            st.selectbox(
                label="行业",
                options=[
                    None,
                    *industries,
                ],
                format_func=lambda value: (
                    value or "全部行业"
                ),
            )
        )

        companies = sorted(
            {
                document.company
                for document in documents
                if document.company
            }
        )

        selected_company = (
            st.selectbox(
                label="公司或机构",
                options=[
                    None,
                    *companies,
                ],
                format_func=lambda value: (
                    value or "全部公司"
                ),
            )
        )

        with st.expander(
            "报告日期筛选"
        ):
            date_column_1, date_column_2 = (
                st.columns(2)
            )

            with date_column_1:
                selected_date_from = (
                    st.date_input(
                        label="开始日期",
                        value=None,
                    )
                )

            with date_column_2:
                selected_date_to = (
                    st.date_input(
                        label="结束日期",
                        value=None,
                    )
                )

        report_date_from = (
            selected_date_from.isoformat()
            if selected_date_from
            is not None
            else None
        )

        report_date_to = (
            selected_date_to.isoformat()
            if selected_date_to
            is not None
            else None
        )

        if registry is None:
            visible_documents = []

        else:
            try:
                visible_documents = (
                    registry.list_documents(
                        project_id=(
                            selected_project_id
                        ),
                        industry=(
                            selected_industry
                        ),
                        company=(
                            selected_company
                        ),
                        report_date_from=(
                            report_date_from
                        ),
                        report_date_to=(
                            report_date_to
                        ),
                    )
                )

            except ValueError as error:
                visible_documents = []
                st.error(str(error))

        # 失败或处理中记录仍然显示在目录，
        # 但只有 indexed 文档可以参与问答。
        indexed_documents = [
            document
            for document in visible_documents
            if (
                document.status
                == STATUS_INDEXED
            )
        ]

        available_sources = [
            document.saved_name
            for document in indexed_documents
        ]

        # 项目或筛选条件变化时，
        # 默认重新选中当前范围内的全部文档。
        scope_signature = (
            selected_project_id,
            selected_industry,
            selected_company,
            report_date_from,
            report_date_to,
        )

        if (
            st.session_state.get(
                "document_scope_signature"
            )
            != scope_signature
        ):
            st.session_state[
                "document_scope_signature"
            ] = scope_signature

            st.session_state[
                "selected_document_sources"
            ] = available_sources

        selected_sources = (
            st.multiselect(
                label="参与问答的资料",
                options=available_sources,
                key=(
                    "selected_document_sources"
                ),
                help=(
                    "只会在选中的 PDF "
                    "证据范围内执行检索。"
                ),
            )
        )

        st.markdown(
            "#### 筛选后的文档"
        )

        if not visible_documents:
            st.info(
                "当前筛选条件下没有文档。"
            )

        else:
            for document in (
                visible_documents
            ):
                label = document.title

                if document.ticker:
                    label += (
                        f" ({document.ticker})"
                    )

                st.markdown(
                    f"**{label}**"
                )

                project = (
                    project_by_id.get(
                        document.project_id
                    )
                )

                project_name = (
                    project.name
                    if project is not None
                    else "未分配项目"
                )

                st.caption(
                    f"{project_name}"
                    f" · {document.document_type}"
                    f" · "
                    f"{document.industry or '行业未填'}"
                    f" · 状态 {document.status}"
                    f" · 文本 "
                    f"{document.text_chunk_count}"
                    f" · 图像 "
                    f"{document.image_chunk_count}"
                )

                if document.report_date:
                    st.caption(
                        "报告日期："
                        f"{document.report_date}"
                    )

                if document.error_message:
                    st.error(
                        document.error_message
                    )


        st.markdown(
            "#### 文档归属管理"
        )

        if not documents:
            st.info(
                "当前没有可以分配的文档。"
            )

        else:
            # 通过 document_id 定位文档，
            # 标题只负责给用户展示。
            document_by_id = {
                document.document_id: document
                for document in documents
            }

            managed_document_id = (
                st.selectbox(
                    label="选择需要调整的文档",
                    options=[
                        document.document_id
                        for document in documents
                    ],
                    format_func=lambda value: (
                        f"{document_by_id[value].title}"
                        f" · "
                        f"{document_by_id[value].saved_name}"
                    ),
                    key=(
                        "managed_document_id"
                    ),
                )
            )

            managed_document = (
                document_by_id[
                    managed_document_id
                ]
            )

            # 切换待管理文档时，把目标项目同步为该文档
            # 当前所属项目，避免用户直接点击保存后误移出项目。
            if (
                st.session_state.get(
                    "managed_document_assignment_signature"
                )
                != managed_document_id
            ):
                st.session_state[
                    "managed_document_assignment_signature"
                ] = managed_document_id
                st.session_state[
                    "target_project_id"
                ] = managed_document.project_id

            current_project = (
                project_by_id.get(
                    managed_document.project_id
                )
            )

            current_project_name = (
                current_project.name
                if current_project is not None
                else "未分配项目"
            )

            st.caption(
                "当前归属："
                f"{current_project_name}"
            )

            target_project_id = (
                st.selectbox(
                    label="调整到",
                    options=project_options,
                    format_func=lambda value: (
                        "未分配项目"
                        if value is None
                        else project_by_id[
                            value
                        ].name
                    ),
                    key=(
                        "target_project_id"
                    ),
                )
            )

            if st.button(
                "保存文档归属",
                use_container_width=True,
            ):
                if registry is None:
                    st.error(
                        "文档注册服务尚未连接。"
                    )

                else:
                    try:
                        registry.assign_document(
                            document_id=(
                                managed_document_id
                            ),
                            project_id=(
                                target_project_id
                            ),
                        )

                        target_project = (
                            project_by_id.get(
                                target_project_id
                            )
                        )

                        target_project_name = (
                            target_project.name
                            if target_project
                            is not None
                            else "未分配项目"
                        )

                        st.session_state[
                            "project_action_notice"
                        ] = (
                            f"已将《"
                            f"{managed_document.title}"
                            f"》调整到："
                            f"{target_project_name}"
                        )

                        st.rerun()

                    except (
                        KeyError,
                        ValueError,
                    ) as error:
                        st.error(str(error))


            with st.expander(
                "处理图表页"
            ):
                st.caption(
                    "输入包含图表、表格或重要示意图的"
                    "PDF物理页码。一次最多处理5页。"
                )

                st.info(
                    "每个尚未处理的页面最多产生"
                    "1次视觉模型调用和1次Embedding调用；"
                    "已有图像证据会直接跳过。"
                )

                # 处理完成后页面会重新运行，
                # 因此使用session_state暂存上一次报告。
                last_visual_report = (
                    st.session_state.pop(
                        "last_visual_ingestion_report",
                        None,
                    )
                )

                if (
                    last_visual_report
                    is not None
                ):
                    if (
                        last_visual_report
                        .document_id
                        == managed_document_id
                    ):
                        render_visual_ingestion_report(
                            last_visual_report
                        )

                visual_page_input = (
                    st.text_input(
                        label="图表页码",
                        placeholder=(
                            "例如：3, 7, 9-10"
                        ),
                        help=(
                            "支持英文逗号、中文逗号"
                            "和连续页码范围。"
                        ),
                        key=(
                            "visual_page_input_"
                            f"{managed_document_id}"
                        ),
                    )
                )

                st.caption(
                    f"当前文档共"
                    f"{managed_document.total_pages}页，"
                    f"已有图像证据"
                    f"{managed_document.image_chunk_count}条。"
                )

                visual_button_disabled = (
                    visual_ingestion_service
                    is None
                    or not visual_page_input.strip()
                )

                if st.button(
                    "生成图像证据",
                    disabled=(
                        visual_button_disabled
                    ),
                    use_container_width=True,
                    key=(
                        "ingest_visual_pages_"
                        f"{managed_document_id}"
                    ),
                ):
                    try:
                        selected_visual_pages = (
                            parse_page_numbers(
                                visual_page_input
                            )
                        )

                        if not selected_visual_pages:
                            raise ValueError(
                                "请至少输入一个图表页码"
                            )

                        with st.spinner(
                            "正在渲染页面、分析图表"
                            "并写入Milvus……"
                        ):
                            visual_report = (
                                visual_ingestion_service
                                .ingest_pages(
                                    document_id=(
                                        managed_document_id
                                    ),
                                    page_numbers=(
                                        selected_visual_pages
                                    ),
                                )
                            )

                        st.session_state[
                            "last_visual_ingestion_report"
                        ] = visual_report

                        # 重新读取Milvus实体数和
                        # SQLite图像证据统计。
                        st.rerun()

                    except (
                        TypeError,
                        ValueError,
                        KeyError,
                        FileNotFoundError,
                    ) as error:
                        st.error(str(error))

                    except Exception as error:
                        st.error(
                            "图表页处理失败，请检查"
                            "视觉模型、Embedding和Milvus。"
                        )

                        print(
                            "Streamlit图表页处理错误："
                            f"{type(error).__name__}: "
                            f"{error}"
                        )

            with st.expander(
                "重新入库与删除"
            ):
                st.caption(
                    "重新入库会重新生成文本向量，"
                    "已有图表证据会被保留。"
                )

                default_reindex_pages = min(
                    max(
                        managed_document
                        .processed_pages,
                        1,
                    ),
                    30,
                )

                reindex_max_pages = (
                    st.number_input(
                        label="重新处理页数",
                        min_value=1,
                        max_value=30,
                        value=(
                            default_reindex_pages
                        ),
                        step=1,
                        key=(
                            "reindex_max_pages_"
                            f"{managed_document_id}"
                        ),
                    )
                )

                if st.button(
                    "重新生成文本证据",
                    disabled=(
                        document_manager is None
                    ),
                    use_container_width=True,
                    key=(
                        "reindex_document_"
                        f"{managed_document_id}"
                    ),
                ):
                    with st.spinner(
                        "正在删除旧文本证据并重新入库……"
                    ):
                        try:
                            reindex_report = (
                                document_manager
                                .reindex_document(
                                    document_id=(
                                        managed_document_id
                                    ),
                                    max_pages=int(
                                        reindex_max_pages
                                    ),
                                )
                            )

                            upload_report = (
                                reindex_report
                                .upload_report
                            )

                            st.session_state[
                                "document_action_notice"
                            ] = (
                                "重新入库完成：删除旧文本证据 "
                                f"{reindex_report.deleted_text_count}"
                                " 条，写入文本证据 "
                                f"{upload_report.inserted_chunks}"
                                " 条。"
                            )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                "重新入库失败："
                                f"{error}"
                            )

                st.divider()

                st.warning(
                    "删除会同时清理Milvus证据、"
                    "SQLite记录、本地PDF和页面图片。"
                )

                delete_confirmed = (
                    st.checkbox(
                        "我确认删除当前文档及其全部证据",
                        key=(
                            "confirm_delete_"
                            f"{managed_document_id}"
                        ),
                    )
                )

                if st.button(
                    "删除当前文档",
                    disabled=(
                        not delete_confirmed
                        or document_manager
                        is None
                    ),
                    use_container_width=True,
                    key=(
                        "delete_document_"
                        f"{managed_document_id}"
                    ),
                ):
                    with st.spinner(
                        "正在清理文档和证据……"
                    ):
                        try:
                            delete_report = (
                                document_manager
                                .delete_document(
                                    managed_document_id
                                )
                            )

                            st.session_state[
                                "document_action_notice"
                            ] = (
                                "文档删除完成："
                                f"{delete_report.saved_name}"
                                "，共清理 "
                                f"{delete_report.deleted_evidence_count}"
                                " 条证据。备份位置："
                                f"{delete_report.backup_directory}"
                            )

                            st.session_state[
                                "document_action_warnings"
                            ] = (
                                delete_report
                                .cleanup_warnings
                            )

                            # 下一次运行时清理已经失效的
                            # 文档选择和问答范围。
                            st.session_state[
                                "reset_document_widgets"
                            ] = True

                            st.rerun()

                        except Exception as error:
                            st.error(
                                "文档删除失败："
                                f"{error}"
                            )

        # SEC官方披露区域从这里开始。
        st.divider()
        st.subheader("SEC官方披露")
        st.caption(
            "输入美股Ticker，查询最近的10-K和10-Q。"
            "查询结果来自SEC官方目录，保存后仍保留原文链接。"
        )

        sec_action_notice = (
            st.session_state.pop(
                "sec_action_notice",
                None,
            )
        )

        research_mode = st.radio(
            label="问答模式",
            options=(
                "仅资料库",
                "综合研究",
            ),
            horizontal=True,
            help=(
                "仅资料库只使用选中的PDF；综合研究还会使用"
                "当前项目中已完成入库的SEC官方披露。"
            ),
        )

        indexed_sec_sources = []
        if (
            research_mode == "综合研究"
            and external_source_registry is not None
        ):
            try:
                indexed_sec_sources = (
                    external_source_registry.list_sources(
                        status=EXTERNAL_STATUS_INDEXED,
                        project_id=selected_project_id,
                    )
                    if selected_project_id is not None
                    else external_source_registry.list_sources(
                        status=EXTERNAL_STATUS_INDEXED
                    )
                )
            except Exception as error:
                st.warning(
                    "读取可检索SEC资料失败："
                    f"{error}"
                )

        external_source_ids = [
            record.source_id
            for record in indexed_sec_sources
        ]
        source_metadata = {
            document.saved_name: RetrievalSourceMetadata(
                source_type="pdf",
                display_title=document.title,
                source_date=document.report_date,
            )
            for document in indexed_documents
            if document.saved_name in selected_sources
        }
        source_metadata.update({
            record.source_id: RetrievalSourceMetadata(
                source_type="sec",
                display_title=(
                    f"{record.ticker} {record.form_type} "
                    f"({record.report_date or record.filing_date})"
                ),
                source_url=record.document_url,
                source_date=(
                    record.report_date
                    or record.filing_date
                ),
            )
            for record in indexed_sec_sources
        })
        query_sources = [
            *selected_sources,
            *external_source_ids,
        ]

        if research_mode == "综合研究":
            st.caption(
                "当前综合范围："
                f"{len(selected_sources)}份PDF + "
                f"{len(external_source_ids)}份SEC正文"
            )

        if sec_action_notice:
            st.success(sec_action_notice)

        if sec_service_error:
            st.warning(
                "SEC服务暂时不可用："
                f"{sec_service_error}"
            )

        with st.form(
            "sec_search_form"
        ):
            sec_ticker = st.text_input(
                label="美股Ticker",
                placeholder="例如：NVDA、TGT",
                key="sec_search_ticker",
            )

            sec_result_limit = (
                st.number_input(
                    label="最多显示披露数量",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                )
            )

            sec_search_submitted = (
                st.form_submit_button(
                    "查询SEC披露",
                    disabled=(
                        sec_client is None
                    ),
                    use_container_width=True,
                )
            )

        if sec_search_submitted:
            # 新查询开始时先清除旧结果，
            # 避免错误发生后继续显示上一次公司。
            st.session_state[
                "sec_search_results"
            ] = []

            try:
                with st.spinner(
                    "正在查询SEC官方目录……"
                ):
                    sec_filings = (
                        sec_client
                        .get_recent_filings(
                            ticker=sec_ticker,
                            limit=int(
                                sec_result_limit
                            ),
                        )
                    )

                st.session_state[
                    "sec_search_results"
                ] = sec_filings

                if not sec_filings:
                    st.info(
                        "没有找到近期10-K或10-Q。"
                    )

            except (
                ValueError,
                LookupError,
                SecEdgarError,
            ) as error:
                st.error(str(error))

            except Exception as error:
                st.error(
                    "SEC查询失败，请稍后重试。"
                )
                print(
                    "Streamlit SEC查询错误："
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        sec_search_results = (
            st.session_state.get(
                "sec_search_results",
                [],
            )
        )

        if external_source_registry is None:
            saved_sec_source_ids: set[str] = set()
            saved_sec_sources = []

        else:
            try:
                saved_sec_sources = (
                    external_source_registry
                    .list_sources()
                )
                saved_sec_source_ids = {
                    record.source_id
                    for record
                    in saved_sec_sources
                }

            except Exception as error:
                saved_sec_sources = []
                saved_sec_source_ids = set()
                st.warning(
                    "读取已保存SEC披露失败："
                    f"{error}"
                )

        if sec_search_results:
            first_company = (
                sec_search_results[0]
                .company
            )

            st.markdown(
                f"**{first_company.name}**"
            )
            st.caption(
                f"Ticker：{first_company.ticker}"
                f" · CIK：{first_company.cik}"
            )

            for filing in sec_search_results:
                report_date_text = (
                    filing.report_date
                    or "未提供"
                )

                with st.container(
                    border=True
                ):
                    st.markdown(
                        f"**{filing.form_type}**"
                        f" · 报告期 "
                        f"{report_date_text}"
                    )
                    st.caption(
                        "提交日期："
                        f"{filing.filing_date}"
                        " · Accession："
                        f"{filing.accession_number}"
                    )

                    link_column, save_column = (
                        st.columns(2)
                    )

                    with link_column:
                        st.link_button(
                            "打开SEC原文",
                            filing.document_url,
                            use_container_width=True,
                        )

                    already_saved = (
                        filing.source_id
                        in saved_sec_source_ids
                    )

                    with save_column:
                        if st.button(
                            (
                                "已经保存"
                                if already_saved
                                else "保存到研究资料"
                            ),
                            disabled=(
                                already_saved
                                or external_source_registry
                                is None
                            ),
                            use_container_width=True,
                            key=(
                                "save_sec_filing_"
                                f"{filing.source_id}"
                            ),
                        ):
                            try:
                                save_result = (
                                    external_source_registry
                                    .save_filing(
                                        filing=filing,
                                        project_id=(
                                            selected_project_id
                                        ),
                                    )
                                )

                                project_name = (
                                    project_by_id[
                                        selected_project_id
                                    ].name
                                    if selected_project_id
                                    is not None
                                    else "未分配项目"
                                )

                                if save_result.created:
                                    notice = (
                                        "已保存SEC披露："
                                        f"{filing.company.ticker} "
                                        f"{filing.form_type}"
                                        f"，归属：{project_name}"
                                    )
                                else:
                                    notice = (
                                        "该SEC披露已经保存。"
                                    )

                                st.session_state[
                                    "sec_action_notice"
                                ] = notice
                                st.rerun()

                            except Exception as error:
                                st.error(
                                    "保存SEC披露失败："
                                    f"{error}"
                                )

        with st.expander(
            "已保存的SEC披露"
        ):
            if selected_project_id is None:
                visible_sec_sources = (
                    saved_sec_sources
                )
            else:
                visible_sec_sources = [
                    record
                    for record
                    in saved_sec_sources
                    if (
                        record.project_id
                        == selected_project_id
                    )
                ]

            if not visible_sec_sources:
                st.info(
                    "当前范围还没有保存SEC披露。"
                )

            for record in visible_sec_sources:
                st.markdown(
                    f"**{record.ticker} "
                    f"{record.form_type}**"
                )
                st.caption(
                    f"{record.company_name}"
                    f" · 报告期 "
                    f"{record.report_date or '未提供'}"
                    f" · 提交 {record.filing_date}"
                )
                status_text = {
                    "saved": "已保存，尚未入库",
                    "indexed": (
                        f"已入库 {record.evidence_count} 条证据"
                    ),
                    "failed": "入库失败，可重试",
                }.get(record.status, record.status)
                st.caption(f"状态：{status_text}")

                link_column, index_column = st.columns(2)
                with link_column:
                    st.link_button(
                        "查看官方原文",
                        record.document_url,
                        use_container_width=True,
                        key=(
                            "open_saved_sec_"
                            f"{record.source_id}"
                        ),
                    )

                with index_column:
                    if st.button(
                        (
                            "正文已入库"
                            if record.status
                            == EXTERNAL_STATUS_INDEXED
                            else "下载正文并入库"
                        ),
                        disabled=(
                            record.status
                            == EXTERNAL_STATUS_INDEXED
                            or external_ingestion_service
                            is None
                        ),
                        use_container_width=True,
                        key=(
                            "index_saved_sec_"
                            f"{record.source_id}"
                        ),
                    ):
                        try:
                            with st.spinner(
                                "正在下载、清洗并索引SEC正文……"
                            ):
                                report = (
                                    external_ingestion_service.ingest(
                                        record.source_id,
                                        max_chunks=500,
                                    )
                                )

                            truncation = (
                                "；已达到500条成本上限"
                                if report.truncated
                                else ""
                            )
                            st.session_state[
                                "sec_action_notice"
                            ] = (
                                "SEC正文入库完成："
                                f"新增{report.inserted_count}条，"
                                f"跳过{report.skipped_count}条"
                                f"{truncation}。"
                            )
                            st.rerun()
                        except Exception as error:
                            st.error(
                                "SEC正文入库失败："
                                f"{error}"
                            )

                if record.error_message:
                    st.warning(
                        f"上次错误：{record.error_message}"
                    )

        # SEC官方披露区域到这里结束。

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

        upload_project_id = (
            st.selectbox(
                label="归属研究项目",
                options=project_options,
                format_func=lambda value: (
                    "暂不分配项目"
                    if value is None
                    else project_by_id[
                        value
                    ].name
                ),
                help=(
                    "入库完成后，将文档直接"
                    "归入所选研究项目。"
                ),
                key="upload_project_id",
            )
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
                    # PDF入库成功后，再更新文档与项目的关系。
                    # UploadReport 中的 document_id 对应
                    # SQLite documents 表的主键。
                    if (
                        registry is not None
                        and upload_project_id
                        is not None
                    ):
                        registry.assign_document(
                            document_id=(
                                upload_report.document_id
                            ),
                            project_id=(
                                upload_project_id
                            ),
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

    # 研究简报使用当前侧边栏已经选定的项目、PDF 和 SEC 范围，
    # 因此无需再维护一套容易不一致的资料选择控件。
    st.subheader("研究简报")
    st.caption(
        "生成固定结构的Markdown简报。每条事实必须通过本轮"
        "证据ID校验，文件保存在data/reports并可直接下载。"
    )

    selected_project = (
        project_by_id.get(selected_project_id)
        if selected_project_id is not None
        else None
    )
    report_project_name = (
        selected_project.name
        if selected_project is not None
        else "跨项目研究"
    )

    with st.form("research_report_form"):
        report_question = st.text_area(
            label="简报研究问题",
            placeholder=(
                "例如：比较所选公司在AI基础设施中的定位、"
                "增长驱动、风险和信息缺口"
            ),
            height=100,
        )
        report_date_value = st.date_input(
            label="简报日期",
            value=date.today(),
            help="该日期只用于标记本次研究成果，不会改写来源日期。",
        )
        st.caption(
            f"当前项目：{report_project_name} · "
            f"当前来源：{len(selected_sources)}份PDF + "
            f"{len(external_source_ids)}份SEC正文"
        )
        generate_report = st.form_submit_button(
            "生成研究简报",
            disabled=(
                research_report_service is None
                or not query_sources
            ),
            use_container_width=True,
        )

    if generate_report:
        try:
            with st.spinner(
                "正在检索证据、生成简报并校验全部引用……"
            ):
                report_result = research_report_service.generate(
                    question=report_question,
                    project_name=report_project_name,
                    report_date=report_date_value.isoformat(),
                    source=query_sources,
                    source_metadata=source_metadata,
                )
            st.session_state["latest_research_report"] = (
                report_result
            )
        except (ValueError, CitationValidationError) as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "研究简报生成失败，请检查模型、网络和证据范围。"
            )
            print(
                "Streamlit研究简报错误："
                f"{type(error).__name__}: {error}"
            )

    latest_report = st.session_state.get(
        "latest_research_report"
    )
    if latest_report is not None:
        with st.expander(
            f"最新简报：{latest_report.title}",
            expanded=True,
        ):
            st.markdown(latest_report.markdown)
            st.caption(
                f"检索证据：{latest_report.retrieved_count}条 · "
                f"有效引用：{len(latest_report.citations)}条 · "
                f"保存位置：{latest_report.output_path}"
            )
            st.download_button(
                "下载Markdown简报",
                data=latest_report.markdown.encode("utf-8"),
                file_name=latest_report.output_path.name,
                mime="text/markdown",
                use_container_width=True,
            )

    st.divider()

    # 初始化聊天记录，只在第一次执行
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 遍历所有聊天消息
    for message in st.session_state.messages:
        render_message(message)

    knowledge_base_ready = (
        answerer is not None
        and entity_count > 0
        and bool(query_sources)
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


    # 综合研究模式会把已索引 SEC source_id 与用户选中的
    # PDF 文件名合并为一次明确的 Milvus 来源过滤。
    source_filter = query_sources
    source_labels = [
        (
            source_metadata[source_name].display_title
            if source_name in source_metadata
            else source_name
        )
        for source_name in query_sources
    ]

    user_message = {
        "role": "user",
        "content": question,
        "source": source_labels,
    }

    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("正在检索证据并生成回答……"):
            try:
                result = answerer.answer(
                    question=question,
                    source=source_filter,
                    source_metadata=source_metadata,
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
