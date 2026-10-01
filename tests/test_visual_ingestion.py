"""测试上传后图表页处理流程。"""

from pathlib import Path

import pytest

from research_kb.document_registry import (
    DocumentMetadata,
    DocumentRegistry,
)
from research_kb.indexer import IndexingReport
from research_kb.page_renderer import RenderedPage
from research_kb.vision_service import (
    VisualDescription,
)
from research_kb.visual_evidence import (
    build_visual_evidence_id,
)
from research_kb.visual_ingestion import (
    VisualIngestionService,
    parse_page_numbers,
)


class FakeVisualStore:
    """模拟证据ID查询和来源统计。"""

    def __init__(
        self,
        existing_ids: set[str] | None = None,
    ) -> None:
        self.existing_ids = existing_ids or set()
        self.id_queries: list[list[str]] = []

    def get_existing_ids(
        self,
        evidence_ids: list[str],
    ) -> set[str]:
        self.id_queries.append(evidence_ids)
        return self.existing_ids.intersection(evidence_ids)

    def get_source_statistics(
        self,
        source: str,
    ) -> dict[str, int]:
        return {
            "total": 14,
            "text": 12,
            "image": 2,
            "max_page": 3,
        }


class FakeVisualIndexer:
    """记录真正进入Embedding流程的页码。"""

    def __init__(self) -> None:
        self.indexed_pages: list[int] = []

    def index_chunks(
        self,
        chunks,
        batch_size: int = 10,
    ) -> IndexingReport:
        self.indexed_pages.extend(chunk.page_number for chunk in chunks)
        return IndexingReport(
            total_chunks=len(chunks),
            inserted_chunks=len(chunks),
            skipped_chunks=0,
        )


class FakeVisionService:
    """第3页失败，其余页面返回固定描述。"""

    def __init__(self) -> None:
        self.called_pages: list[int] = []

    def describe_page(
        self,
        source: str,
        page_number: int,
        image_path: str | Path,
    ) -> VisualDescription:
        self.called_pages.append(page_number)

        if page_number == 3:
            raise RuntimeError("模拟视觉模型失败")

        return VisualDescription(
            source=source,
            page_number=page_number,
            image_path=Path(image_path),
            description=(f"第{page_number}页图表描述"),
        )


def create_indexed_document(
    tmp_path: Path,
) -> tuple[
    DocumentRegistry,
    str,
    Path,
]:
    """创建已完成文本入库的临时文档。"""
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()

    pdf_path = raw_directory / "test_visual.pdf"
    pdf_path.write_bytes(b"fake-pdf")

    registry = DocumentRegistry(tmp_path / "registry.db")
    document, _ = registry.register_document(
        document_hash="f" * 64,
        original_name=pdf_path.name,
        saved_name=pdf_path.name,
        metadata=DocumentMetadata(
            title="视觉测试",
            document_type="年报",
        ),
        total_pages=4,
    )
    registry.mark_indexed(
        document_id=document.document_id,
        total_pages=4,
        processed_pages=4,
        text_chunk_count=12,
        image_chunk_count=0,
    )

    return (
        registry,
        document.document_id,
        raw_directory,
    )


def test_parse_page_numbers() -> None:
    """单页、范围、中文逗号和去重都能正确解析。"""
    assert parse_page_numbers("3，7-9") == [3, 7, 8, 9]
    assert parse_page_numbers("3, 3, 2-4") == [3, 2, 4]
    assert parse_page_numbers("") == []

    for invalid_value in (
        "3,,4",
        "7-3",
        "0",
        "a",
        "1-6",
    ):
        with pytest.raises(ValueError):
            parse_page_numbers(invalid_value)


def test_visual_ingestion_skips_and_continues(
    tmp_path: Path,
) -> None:
    """重复页跳过、单页失败继续，并同步实际图像数。"""
    (
        registry,
        document_id,
        raw_directory,
    ) = create_indexed_document(tmp_path)

    existing_id = build_visual_evidence_id(
        source="test_visual.pdf",
        page_number=1,
    )
    store = FakeVisualStore({existing_id})
    indexer = FakeVisualIndexer()
    vision_service = FakeVisionService()
    image_directory = tmp_path / "images"
    image_directory.mkdir()

    def fake_renderer(
        pdf_path: str | Path,
        page_number: int,
    ) -> RenderedPage:
        image_path = image_directory / f"page_{page_number:04d}.png"
        image_path.write_bytes(b"png")
        return RenderedPage(
            source=Path(pdf_path).name,
            page_number=page_number,
            image_path=image_path,
            width=100,
            height=100,
        )

    service = VisualIngestionService(
        store=store,
        indexer=indexer,
        vision_service=vision_service,
        registry=registry,
        raw_data_dir=raw_directory,
        page_renderer=fake_renderer,
    )

    report = service.ingest_pages(
        document_id=document_id,
        page_numbers=[1, 2, 3],
    )

    assert report.model_calls == 2
    assert report.inserted_chunks == 1
    assert report.skipped_chunks == 1
    assert report.failed_pages == (3,)
    assert vision_service.called_pages == [
        2,
        3,
    ]
    assert indexer.indexed_pages == [2]

    updated = registry.get_by_id(document_id)
    assert updated is not None
    assert updated.image_chunk_count == 2
    assert updated.text_chunk_count == 12
    assert updated.status == "indexed"


def test_out_of_range_page_has_no_external_calls(
    tmp_path: Path,
) -> None:
    """越界页在查询Milvus和调用视觉模型前失败。"""
    (
        registry,
        document_id,
        raw_directory,
    ) = create_indexed_document(tmp_path)

    store = FakeVisualStore()
    indexer = FakeVisualIndexer()
    vision_service = FakeVisionService()

    def unexpected_renderer(
        pdf_path: str | Path,
        page_number: int,
    ) -> RenderedPage:
        raise AssertionError("越界时不应调用渲染器")

    service = VisualIngestionService(
        store=store,
        indexer=indexer,
        vision_service=vision_service,
        registry=registry,
        raw_data_dir=raw_directory,
        page_renderer=unexpected_renderer,
    )

    with pytest.raises(
        ValueError,
        match="越界",
    ):
        service.ingest_pages(
            document_id=document_id,
            page_numbers=[5],
        )

    assert store.id_queries == []
    assert vision_service.called_pages == []
    assert indexer.indexed_pages == []
