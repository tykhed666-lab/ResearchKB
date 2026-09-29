"""处理用户选择的PDF图表页并生成图像证据。"""

from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path

from research_kb.document_registry import (
    DocumentRegistry,
    STATUS_INDEXED,
)
from research_kb.indexer import (
    EvidenceIndexer,
)
from research_kb.milvus_store import (
    MilvusStore,
)
from research_kb.page_renderer import (
    RenderedPage,
    get_page_image_path,
    render_pdf_page,
)
from research_kb.upload_service import (
    RAW_DATA_DIR,
)
from research_kb.vision_service import (
    VisionService,
)
from research_kb.visual_evidence import (
    build_visual_evidence_chunk,
    build_visual_evidence_id,
)


# 一次最多调用视觉模型处理5页，
# 防止误输入大量页码产生过多费用。
MAX_VISUAL_PAGES_PER_REQUEST = 5


# Callable表示“可以被调用的对象”。
# 这个类型描述一个接收PDF路径和页码、
# 返回RenderedPage的页面渲染函数。
PageRenderer = Callable[
    [str | Path, int],
    RenderedPage,
]


@dataclass(frozen=True, slots=True)
class VisualPageResult:
    """记录一张图表页的处理结果。"""

    page_number: int

    # inserted：本次新入库；
    # skipped：Milvus已经存在；
    # failed：该页面处理失败。
    status: str

    evidence_id: str
    image_path: Path | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class VisualIngestionReport:
    """记录一次多页视觉处理的汇总结果。"""

    document_id: str
    source: str
    requested_pages: tuple[int, ...]

    # 实际请求视觉模型的次数。
    # 已存在的图像证据不会计入。
    model_calls: int

    inserted_chunks: int
    skipped_chunks: int
    failed_pages: tuple[int, ...]
    page_results: tuple[
        VisualPageResult,
        ...,
    ]


def parse_page_numbers(
    value: str,
) -> list[int]:
    """解析用户输入的图表页码。

    支持以下格式：
    - 3, 7, 9
    - 3, 7-9
    - 中文逗号：3，7-9

    返回去重后的页码，并保留用户输入顺序。
    """
    cleaned_value = (
        value.strip()
        .replace("，", ",")
    )

    if not cleaned_value:
        return []

    page_numbers: list[int] = []

    # 按照','逐段处理
    for raw_part in cleaned_value.split(","):
        part = raw_part.strip()

        if not part:
            raise ValueError(
                "页码之间不能出现空项"
            )

        if "-" in part:
            range_parts = [
                item.strip()
                for item in part.split("-")
            ]

            if len(range_parts) != 2:
                raise ValueError(
                    "页码范围格式不正确："
                    f"{part}"
                )

            try:
                start_page = int(
                    range_parts[0]
                )
                end_page = int(
                    range_parts[1]
                )
            except ValueError as error:
                raise ValueError(
                    "页码必须是整数："
                    f"{part}"
                ) from error

            if (
                start_page <= 0
                or end_page <= 0
            ):
                raise ValueError(
                    "页码必须大于0"
                )

            if start_page > end_page:
                raise ValueError(
                    "页码范围起点不能大于终点："
                    f"{part}"
                )

            page_numbers.extend(
                range(
                    start_page,
                    end_page + 1,
                )
            )

        else:
            try:
                page_number = int(part)
            except ValueError as error:
                raise ValueError(
                    "页码必须是整数："
                    f"{part}"
                ) from error

            if page_number <= 0:
                raise ValueError(
                    "页码必须大于0"
                )

            page_numbers.append(
                page_number
            )

    # dict会保留插入顺序，
    # 因此可以同时完成去重和顺序保留。
    unique_page_numbers = list(
        dict.fromkeys(page_numbers)
    )

    if (
        len(unique_page_numbers)
        > MAX_VISUAL_PAGES_PER_REQUEST
    ):
        raise ValueError(
            "一次最多处理"
            f"{MAX_VISUAL_PAGES_PER_REQUEST}"
            "个不重复页面"
        )

    return unique_page_numbers


class VisualIngestionService:
    """协调图表页渲染、描述、入库和统计更新。"""

    def __init__(
        self,
        store: MilvusStore,
        indexer: EvidenceIndexer,
        vision_service: VisionService,
        registry: DocumentRegistry,
        raw_data_dir: str | Path = (
            RAW_DATA_DIR
        ),
        page_renderer: PageRenderer = (
            render_pdf_page
        ),
    ) -> None:
        """保存视觉处理需要的各项服务。

        page_renderer作为参数传入，
        方便自动化测试时替换成假渲染函数。
        """
        self.store = store
        self.indexer = indexer
        self.vision_service = (
            vision_service
        )
        self.registry = registry
        self.raw_data_dir = Path(
            raw_data_dir
        )
        self.page_renderer = (
            page_renderer
        )

    def ingest_pages(
        self,
        document_id: str,
        page_numbers: Sequence[int],
    ) -> VisualIngestionReport:
        """处理一份文档中用户选择的图表页。

        单页失败不会停止后续页面，也不会把文档
        原有的文本入库状态修改为failed。
        """
        cleaned_document_id = (
            document_id.strip()
        )

        if not cleaned_document_id:
            raise ValueError(
                "document_id不能为空"
            )

        if isinstance(
            page_numbers,
            (str, bytes),
        ):
            raise TypeError(
                "page_numbers必须是整数序列"
            )

        normalized_pages: list[int] = []

        for page_number in page_numbers:
            # bool是int的子类，但True/False显然不是页码，
            # 因此需要单独排除。
            if (
                isinstance(
                    page_number,
                    bool,
                )
                or not isinstance(
                    page_number,
                    int,
                )
            ):
                raise TypeError(
                    "页码必须是整数"
                )

            if page_number <= 0:
                raise ValueError(
                    "页码必须大于0"
                )

            normalized_pages.append(
                page_number
            )

        # 再次去重，避免其他调用者绕过
        # parse_page_numbers直接传入重复值。
        unique_pages = list(
            dict.fromkeys(
                normalized_pages
            )
        )

        if not unique_pages:
            raise ValueError(
                "请至少选择一个图表页"
            )

        if (
            len(unique_pages)
            > MAX_VISUAL_PAGES_PER_REQUEST
        ):
            raise ValueError(
                "一次最多处理"
                f"{MAX_VISUAL_PAGES_PER_REQUEST}"
                "个不重复页面"
            )

        record = self.registry.get_by_id(
            cleaned_document_id
        )

        if record is None:
            raise KeyError(
                "找不到文档："
                f"{cleaned_document_id}"
            )

        if record.status != STATUS_INDEXED:
            raise ValueError(
                "只有文本入库成功的文档"
                "才能继续处理图表页"
            )

        invalid_pages = [
            page_number
            for page_number in unique_pages
            if (
                page_number
                > record.total_pages
            )
        ]

        # 在调用渲染器和视觉模型前统一检查，
        # 避免部分页面已经产生费用后才发现越界。
        if invalid_pages:
            invalid_text = "、".join(
                str(page_number)
                for page_number
                in invalid_pages
            )

            raise ValueError(
                f"PDF共{record.total_pages}页，"
                f"以下页码越界：{invalid_text}"
            )

        safe_saved_name = Path(
            record.saved_name
        ).name

        pdf_path = (
            self.raw_data_dir
            / safe_saved_name
        )

        if not pdf_path.is_file():
            raise FileNotFoundError(
                "找不到文档对应的本地PDF："
                f"{pdf_path}"
            )

        page_results: list[
            VisualPageResult
        ] = []

        model_calls = 0
        inserted_chunks = 0
        skipped_chunks = 0
        failed_pages: list[int] = []

        for page_number in unique_pages:
            evidence_id = (
                build_visual_evidence_id(
                    source=record.saved_name,
                    page_number=page_number,
                )
            )

            expected_image_path = (
                get_page_image_path(
                    source=record.saved_name,
                    page_number=page_number,
                )
            )

            existing_ids = (
                self.store.get_existing_ids(
                    [evidence_id]
                )
            )

            if evidence_id in existing_ids:
                # 已存在的图像证据不再调用视觉模型。
                # 如果本地原图丢失，只重新渲染图片。
                image_path: Path | None = (
                    expected_image_path
                    if (
                        expected_image_path
                        .is_file()
                    )
                    else None
                )
                restore_error = None

                if image_path is None:
                    try:
                        rendered_page = (
                            self.page_renderer(
                                pdf_path,
                                page_number,
                            )
                        )
                        image_path = (
                            rendered_page
                            .image_path
                        )

                    except Exception as error:
                        restore_error = (
                            "证据已经存在，但页面"
                            "图片恢复失败："
                            f"{type(error).__name__}: "
                            f"{error}"
                        )

                skipped_chunks += 1

                page_results.append(
                    VisualPageResult(
                        page_number=(
                            page_number
                        ),
                        status="skipped",
                        evidence_id=evidence_id,
                        image_path=image_path,
                        error_message=(
                            restore_error
                        ),
                    )
                )

                continue

            try:
                rendered_page = (
                    self.page_renderer(
                        pdf_path,
                        page_number,
                    )
                )

                # 只统计真正发送给视觉模型的调用。
                # 即使模型调用抛出异常，也已经发生了一次请求。
                model_calls += 1

                visual_description = (
                    self.vision_service
                    .describe_page(
                        source=(
                            record.saved_name
                        ),
                        page_number=(
                            page_number
                        ),
                        image_path=(
                            rendered_page
                            .image_path
                        ),
                    )
                )

                chunk = (
                    build_visual_evidence_chunk(
                        source=(
                            record.saved_name
                        ),
                        page_number=(
                            page_number
                        ),
                        description=(
                            visual_description
                            .description
                        ),
                    )
                )

                indexing_report = (
                    self.indexer.index_chunks(
                        [chunk],
                        batch_size=1,
                    )
                )

                inserted_chunks += (
                    indexing_report
                    .inserted_chunks
                )
                skipped_chunks += (
                    indexing_report
                    .skipped_chunks
                )

                page_status = (
                    "inserted"
                    if (
                        indexing_report
                        .inserted_chunks
                        > 0
                    )
                    else "skipped"
                )

                page_results.append(
                    VisualPageResult(
                        page_number=(
                            page_number
                        ),
                        status=page_status,
                        evidence_id=evidence_id,
                        image_path=(
                            rendered_page
                            .image_path
                        ),
                        error_message=None,
                    )
                )

            except Exception as error:
                # 单页失败只记录结果，继续处理剩余页面。
                failed_pages.append(
                    page_number
                )

                page_results.append(
                    VisualPageResult(
                        page_number=(
                            page_number
                        ),
                        status="failed",
                        evidence_id=evidence_id,
                        image_path=(
                            expected_image_path
                            if (
                                expected_image_path
                                .is_file()
                            )
                            else None
                        ),
                        error_message=(
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    )
                )

        # 以Milvus实际查询结果作为最终统计，
        # 避免重复页和部分失败导致SQLite计数不准确。
        source_statistics = (
            self.store
            .get_source_statistics(
                record.saved_name
            )
        )

        self.registry.update_image_chunk_count(
            document_id=(
                record.document_id
            ),
            image_chunk_count=(
                source_statistics["image"]
            ),
        )

        return VisualIngestionReport(
            document_id=record.document_id,
            source=record.saved_name,
            requested_pages=tuple(
                unique_pages
            ),
            model_calls=model_calls,
            inserted_chunks=(
                inserted_chunks
            ),
            skipped_chunks=(
                skipped_chunks
            ),
            failed_pages=tuple(
                failed_pages
            ),
            page_results=tuple(
                page_results
            ),
        )
