"""测试外部研究资料的 SQLite 登记簿。"""

from pathlib import Path
import sqlite3

import pytest

from research_kb.document_registry import (
    DocumentRegistry,
)
from research_kb.external_source_registry import (
    EXTERNAL_STATUS_SAVED,
    ExternalSourceRegistry,
)
from research_kb.sec_edgar import (
    SecCompany,
    SecFiling,
)


def build_filing(
    ticker: str = "TGT",
    accession_number: str = (
        "0000027419-26-000016"
    ),
    filing_date: str = "2026-03-11",
) -> SecFiling:
    """构造测试使用的 SEC 披露。"""
    accession_directory = (
        accession_number.replace("-", "")
    )
    cik = "0000027419"

    return SecFiling(
        source_id=(
            f"sec_{cik}_{accession_directory}"
        ),
        company=SecCompany(
            ticker=ticker,
            cik=cik,
            name="TARGET CORP",
        ),
        form_type="10-K",
        accession_number=accession_number,
        filing_date=filing_date,
        report_date="2026-01-31",
        primary_document="tgt-20260131.htm",
        document_url=(
            "https://www.sec.gov/Archives/"
            f"edgar/data/27419/{accession_directory}/"
            "tgt-20260131.htm"
        ),
        index_url=(
            "https://www.sec.gov/Archives/"
            f"edgar/data/27419/{accession_directory}/"
            f"{accession_number}-index.html"
        ),
    )


def create_registries(
    database_path: Path,
) -> tuple[
    DocumentRegistry,
    ExternalSourceRegistry,
]:
    """创建共享同一临时数据库的两个登记簿。"""
    return (
        DocumentRegistry(database_path),
        ExternalSourceRegistry(database_path),
    )


def test_save_and_skip_duplicate(
    tmp_path: Path,
) -> None:
    """同一 SEC 披露重复保存时不创建新记录。"""
    _, registry = create_registries(
        tmp_path / "research.db"
    )
    filing = build_filing()

    first = registry.save_filing(filing)
    second = registry.save_filing(filing)

    assert first.created is True
    assert second.created is False
    assert first.record == second.record
    assert first.record.status == (
        EXTERNAL_STATUS_SAVED
    )
    assert registry.count_sources() == 1
    assert registry.get_by_id(
        filing.source_id
    ) == first.record
    assert registry.get_by_accession_number(
        filing.accession_number
    ) == first.record


def test_save_filing_in_project_and_filter(
    tmp_path: Path,
) -> None:
    """外部资料可以归入项目并按项目和Ticker筛选。"""
    documents, registry = create_registries(
        tmp_path / "research.db"
    )
    project = documents.create_project(
        name="消费零售",
        description="零售公司测试",
    )
    filing = build_filing()

    result = registry.save_filing(
        filing,
        project_id=project.project_id,
    )

    assert result.record.project_id == (
        project.project_id
    )
    assert registry.list_sources(
        ticker="tgt"
    ) == [result.record]
    assert registry.list_sources(
        project_id=project.project_id
    ) == [result.record]
    assert registry.list_sources(
        status="saved"
    ) == [result.record]
    assert registry.list_sources(
        ticker="NVDA"
    ) == []


def test_unknown_project_rolls_back(
    tmp_path: Path,
) -> None:
    """不存在的项目不能留下半条外部资料记录。"""
    _, registry = create_registries(
        tmp_path / "research.db"
    )

    with pytest.raises(
        KeyError,
        match="找不到研究项目",
    ):
        registry.save_filing(
            build_filing(),
            project_id="missing-project",
        )

    assert registry.count_sources() == 0


def test_invalid_date_is_rejected(
    tmp_path: Path,
) -> None:
    """错误日期格式应在写入数据库前被拒绝。"""
    _, registry = create_registries(
        tmp_path / "research.db"
    )
    filing = build_filing(
        filing_date="2026/03/11"
    )

    with pytest.raises(
        ValueError,
        match="提交日期必须使用 YYYY-MM-DD",
    ):
        registry.save_filing(filing)

    assert registry.count_sources() == 0


def test_invalid_status_filter_is_rejected(
    tmp_path: Path,
) -> None:
    """不支持的状态不能进入 SQL 筛选条件。"""
    _, registry = create_registries(
        tmp_path / "research.db"
    )

    with pytest.raises(
        ValueError,
        match="不支持的外部资料状态",
    ):
        registry.list_sources(
            status="unknown"
        )


def test_existing_table_gets_day12_columns(
    tmp_path: Path,
) -> None:
    """第 11 天创建的旧表应幂等补齐正文入库字段。"""
    database_path = tmp_path / "old.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE external_sources (
                source_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                ticker TEXT NOT NULL,
                cik TEXT NOT NULL,
                company_name TEXT NOT NULL,
                form_type TEXT NOT NULL,
                report_date TEXT,
                filing_date TEXT NOT NULL,
                accession_number TEXT NOT NULL UNIQUE,
                primary_document TEXT NOT NULL,
                document_url TEXT NOT NULL,
                index_url TEXT NOT NULL,
                collected_at TEXT NOT NULL,
                status TEXT NOT NULL,
                project_id TEXT
            );
            """
        )

    ExternalSourceRegistry(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(external_sources)"
            )
        }

    assert {
        "local_path",
        "evidence_count",
        "indexed_at",
        "error_message",
    }.issubset(columns)
