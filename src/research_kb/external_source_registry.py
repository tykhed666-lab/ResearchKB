"""使用 SQLite 登记用户保存的外部研究资料。"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

from research_kb.document_registry import (
    CREATE_PROJECTS_SQL,
)
from research_kb.sec_edgar import (
    SecFiling,
)
from research_kb.settings import (
    DOCUMENT_DATABASE_PATH,
)

EXTERNAL_STATUS_SAVED: Final = "saved"
EXTERNAL_STATUS_INDEXED: Final = "indexed"
EXTERNAL_STATUS_FAILED: Final = "failed"

VALID_EXTERNAL_STATUSES: Final = frozenset(
    {
        EXTERNAL_STATUS_SAVED,
        EXTERNAL_STATUS_INDEXED,
        EXTERNAL_STATUS_FAILED,
    }
)


@dataclass(
    frozen=True,
    slots=True,
)
class ExternalSourceRecord:
    """表示 SQLite 中的一条外部资料记录."""

    source_id: str
    provider: str
    ticker: str
    cik: str
    company_name: str
    form_type: str
    report_date: str | None
    filing_date: str
    accession_number: str
    primary_document: str
    document_url: str
    index_url: str
    collected_at: str
    status: str
    project_id: str | None
    local_path: str | None
    evidence_count: int
    indexed_at: str | None
    error_message: str | None


@dataclass(
    frozen=True,
    slots=True,
)
class ExternalSaveResult:
    """记录一次外部资料保存操作。"""

    record: ExternalSourceRecord

    # True表示本次创建了新记录；
    # False表示相同披露已经存在。
    created: bool


CREATE_EXTERNAL_SOURCES_SQL: Final = """
CREATE TABLE IF NOT EXISTS external_sources (
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
    status TEXT NOT NULL CHECK(
        status IN (
            'saved',
            'indexed',
            'failed'
        )
    ),
    project_id TEXT
        REFERENCES projects(project_id),
    local_path TEXT,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS
    idx_external_sources_ticker
ON external_sources(ticker);

CREATE INDEX IF NOT EXISTS
    idx_external_sources_status
ON external_sources(status);

CREATE INDEX IF NOT EXISTS
    idx_external_sources_project_id
ON external_sources(project_id);

CREATE INDEX IF NOT EXISTS
    idx_external_sources_filing_date
ON external_sources(filing_date);
"""


class ExternalSourceRegistry:
    """管理保存在 SQLite 中的外部资料。"""

    def __init__(
        self,
        database_path: (str | Path) = DOCUMENT_DATABASE_PATH,
    ) -> None:
        """保存数据库地址并初始化数据表。"""
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    @contextmanager
    def _connect(
        self,
    ) -> Iterator[sqlite3.Connection]:
        """创建带事务管理的 SQLite 连接。"""
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
        )

        connection.row_factory = sqlite3.Row

        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            # 提交
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def _initialize_database(
        self,
    ) -> None:
        """创建项目表和外部资料表。

        外部资料可以归属于已有研究项目，
        因此先确保 projects 表存在。
        """
        with self._connect() as connection:
            connection.executescript(CREATE_PROJECTS_SQL)

            connection.executescript(CREATE_EXTERNAL_SOURCES_SQL)

            # 旧数据库通过幂等迁移补齐第 12 天字段。
            existing_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(external_sources)"
                ).fetchall()
            }
            migrations = {
                "local_path": "TEXT",
                "evidence_count": ("INTEGER NOT NULL DEFAULT 0"),
                "indexed_at": "TEXT",
                "error_message": "TEXT",
            }
            for column_name, definition in migrations.items():
                if column_name not in existing_columns:
                    connection.execute(
                        "ALTER TABLE external_sources "
                        f"ADD COLUMN {column_name} {definition}"
                    )

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row | None,
    ) -> ExternalSourceRecord | None:
        """将 SQLite Row 转换为业务对象。"""
        if row is None:
            return None

        return ExternalSourceRecord(**dict(row))

    @staticmethod
    def _validate_iso_date(
        value: str,
        field_name: str,
    ) -> str:
        """校验并返回 YYYY-MM-DD 日期。"""
        cleaned = value.strip()

        try:
            date.fromisoformat(cleaned)

        except ValueError as error:
            raise ValueError(f"{field_name}必须使用 YYYY-MM-DD") from error

        return cleaned

    def get_by_id(
        self,
        source_id: str,
    ) -> ExternalSourceRecord | None:
        """根据来源 ID 查询外部资料。"""
        cleaned_source_id = source_id.strip()

        if not cleaned_source_id:
            raise ValueError("source_id不能为空")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM external_sources
                WHERE source_id = ?
                """,
                (cleaned_source_id,),
            ).fetchone()

        return self._row_to_record(row)

    def get_by_accession_number(
        self,
        accession_number: str,
    ) -> ExternalSourceRecord | None:
        """根据 SEC accession number 查询资料。"""
        cleaned_accession = accession_number.strip()

        if not cleaned_accession:
            raise ValueError("accession_number不能为空")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM external_sources
                WHERE accession_number = ?
                """,
                (cleaned_accession,),
            ).fetchone()

        return self._row_to_record(row)

    def save_filing(
        self,
        filing: SecFiling,
        project_id: str | None = None,
    ) -> ExternalSaveResult:
        """保存一份 SEC 披露。

        相同 source_id 或 accession number
        已存在时不会重复创建记录。

        Args:
            filing:
                SecEdgarClient 返回的披露。
            project_id:
                可选的研究项目 ID。
        """
        existing = self.get_by_id(filing.source_id)

        if existing is not None:
            return ExternalSaveResult(
                record=existing,
                created=False,
            )

        existing_accession = self.get_by_accession_number(filing.accession_number)

        if existing_accession is not None:
            return ExternalSaveResult(
                record=existing_accession,
                created=False,
            )

        filing_date = self._validate_iso_date(
            filing.filing_date,
            "提交日期",
        )

        report_date: str | None = None

        if filing.report_date:
            report_date = self._validate_iso_date(
                filing.report_date,
                "报告日期",
            )

        cleaned_project_id = project_id.strip() if project_id else None

        collected_at = datetime.now(UTC).isoformat(timespec="seconds")

        record = ExternalSourceRecord(
            source_id=filing.source_id,
            provider="sec_edgar",
            ticker=(filing.company.ticker),
            cik=filing.company.cik,
            company_name=(filing.company.name),
            form_type=filing.form_type,
            report_date=report_date,
            filing_date=filing_date,
            accession_number=(filing.accession_number),
            primary_document=(filing.primary_document),
            document_url=(filing.document_url),
            index_url=filing.index_url,
            collected_at=collected_at,
            status=(EXTERNAL_STATUS_SAVED),
            project_id=(cleaned_project_id),
            local_path=None,
            evidence_count=0,
            indexed_at=None,
            error_message=None,
        )

        try:
            with self._connect() as connection:
                if cleaned_project_id is not None:
                    project = connection.execute(
                        """
                            SELECT project_id
                            FROM projects
                            WHERE project_id = ?
                            """,
                        (cleaned_project_id,),
                    ).fetchone()

                    if project is None:
                        raise KeyError(f"找不到研究项目：{cleaned_project_id}")

                connection.execute(
                    """
                    INSERT INTO external_sources (
                        source_id,
                        provider,
                        ticker,
                        cik,
                        company_name,
                        form_type,
                        report_date,
                        filing_date,
                        accession_number,
                        primary_document,
                        document_url,
                        index_url,
                        collected_at,
                        status,
                        project_id
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?
                    )
                    """,
                    (
                        record.source_id,
                        record.provider,
                        record.ticker,
                        record.cik,
                        record.company_name,
                        record.form_type,
                        record.report_date,
                        record.filing_date,
                        record.accession_number,
                        record.primary_document,
                        record.document_url,
                        record.index_url,
                        record.collected_at,
                        record.status,
                        record.project_id,
                    ),
                )

        except sqlite3.IntegrityError:
            # 极少数情况下，另一个请求可能在
            # 前面的查询和INSERT之间保存了同一记录。
            duplicate = self.get_by_accession_number(filing.accession_number)

            if duplicate is None:
                raise

            return ExternalSaveResult(
                record=duplicate,
                created=False,
            )

        return ExternalSaveResult(
            record=record,
            created=True,
        )

    def mark_indexed(
        self,
        source_id: str,
        local_path: str | Path,
        evidence_count: int,
    ) -> ExternalSourceRecord:
        """记录外部正文已经下载并完成向量入库。"""
        if evidence_count <= 0:
            raise ValueError("evidence_count 必须大于 0")

        path = Path(local_path).resolve()
        indexed_at = datetime.now(UTC).isoformat(timespec="seconds")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE external_sources
                SET status = ?, local_path = ?,
                    evidence_count = ?, indexed_at = ?,
                    error_message = NULL
                WHERE source_id = ?
                """,
                (
                    EXTERNAL_STATUS_INDEXED,
                    str(path),
                    evidence_count,
                    indexed_at,
                    source_id.strip(),
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"找不到外部资料：{source_id}")

        record = self.get_by_id(source_id)
        if record is None:
            raise RuntimeError("更新后的外部资料不存在")
        return record

    def mark_failed(
        self,
        source_id: str,
        error_message: str,
        local_path: str | Path | None = None,
    ) -> ExternalSourceRecord:
        """记录下载、解析或入库失败，供页面继续重试。"""
        cleaned_error = error_message.strip()
        if not cleaned_error:
            raise ValueError("错误信息不能为空")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE external_sources
                SET status = ?, error_message = ?,
                    local_path = COALESCE(?, local_path)
                WHERE source_id = ?
                """,
                (
                    EXTERNAL_STATUS_FAILED,
                    cleaned_error[:1000],
                    (
                        str(Path(local_path).resolve())
                        if local_path is not None
                        else None
                    ),
                    source_id.strip(),
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"找不到外部资料：{source_id}")

        record = self.get_by_id(source_id)
        if record is None:
            raise RuntimeError("更新后的外部资料不存在")
        return record

    def list_sources(
        self,
        ticker: str | None = None,
        status: str | None = None,
        project_id: str | None = None,
    ) -> list[ExternalSourceRecord]:
        """按照可选条件列出外部资料。"""
        conditions: list[str] = []
        parameters: list[str] = []

        if ticker is not None:
            cleaned_ticker = ticker.strip().upper()

            if not cleaned_ticker:
                raise ValueError("ticker不能为空字符串")

            conditions.append("ticker = ?")
            parameters.append(cleaned_ticker)

        if status is not None:
            cleaned_status = status.strip().lower()

            if cleaned_status not in VALID_EXTERNAL_STATUSES:
                raise ValueError(f"不支持的外部资料状态：{status}")

            conditions.append("status = ?")
            parameters.append(cleaned_status)

        if project_id is not None:
            cleaned_project_id = project_id.strip()

            if not cleaned_project_id:
                raise ValueError("project_id不能为空字符串")

            conditions.append("project_id = ?")
            parameters.append(cleaned_project_id)

        sql = "SELECT * FROM external_sources"

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY filing_date DESC, source_id ASC"

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        return [ExternalSourceRecord(**dict(row)) for row in rows]

    def count_sources(
        self,
    ) -> int:
        """返回已保存的外部资料总数。"""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM external_sources
                """
            ).fetchone()

        if row is None:
            return 0

        return int(row["total"])
