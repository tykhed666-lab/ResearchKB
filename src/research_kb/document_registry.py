"""使用 SQLite 管理文档元数据和处理状态。"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

from research_kb.settings import DOCUMENT_DATABASE_PATH

STATUS_SAVED: Final = "saved"
STATUS_PROCESSING: Final = "processing"
STATUS_INDEXED: Final = "indexed"
STATUS_FAILED: Final = "failed"
VALID_DOCUMENT_STATUSES: Final = frozenset(
    {
        STATUS_SAVED,
        STATUS_PROCESSING,
        STATUS_INDEXED,
        STATUS_FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """用户为研究资料填写的业务信息。"""

    title: str
    document_type: str
    company: str | None = None
    ticker: str | None = None
    industry: str | None = None
    report_date: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """SQLite 中一条完整的文档记录。"""

    document_id: str
    document_hash: str
    original_name: str
    saved_name: str
    title: str
    company: str | None
    ticker: str | None
    industry: str | None
    document_type: str
    report_date: str | None
    uploaded_at: str
    total_pages: int
    processed_pages: int
    text_chunk_count: int
    image_chunk_count: int
    status: str
    error_message: str | None
    # 文档所属研究项目；旧文档迁移后暂时为空。
    project_id: str | None


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    """表示一个轻量研究项目。"""

    project_id: str
    name: str
    description: str | None
    created_at: str


CREATE_PROJECTS_SQL: Final = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL
        UNIQUE COLLATE NOCASE,
    description TEXT,
    created_at TEXT NOT NULL
);
"""

CREATE_DOCUMENTS_SQL: Final = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    document_hash TEXT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    saved_name TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    company TEXT,
    ticker TEXT,
    industry TEXT,
    document_type TEXT NOT NULL,
    report_date TEXT,
    uploaded_at TEXT NOT NULL,
    total_pages INTEGER NOT NULL DEFAULT 0 CHECK(total_pages >= 0),
    processed_pages INTEGER NOT NULL DEFAULT 0 CHECK(processed_pages >= 0),
    text_chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(text_chunk_count >= 0),
    image_chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(image_chunk_count >= 0),
    status TEXT NOT NULL CHECK(
        status IN ('saved', 'processing', 'indexed', 'failed')
    ),
    error_message TEXT,
    project_id TEXT
        REFERENCES projects(project_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company);
CREATE INDEX IF NOT EXISTS idx_documents_ticker ON documents(ticker);
CREATE INDEX IF NOT EXISTS idx_documents_industry ON documents(industry);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""


class DocumentRegistry:
    """负责 SQLite 文档记录的初始化、查询和状态更新。"""

    def __init__(
        self,
        database_path: str | Path = DOCUMENT_DATABASE_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connect(
        self,
    ) -> Iterator[sqlite3.Connection]:
        """创建连接，并在事务结束后保证关闭数据库。"""
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            # yield 把连接临时交给 with 代码块使用。
            yield connection

            # 没有异常时提交本次修改。
            connection.commit()

        except Exception:
            # 发生异常时撤销尚未提交的修改。
            connection.rollback()
            raise

        finally:
            # sqlite3.Connection 自带的 with 只负责事务，
            # 不会自动关闭连接，因此需要显式 close。
            connection.close()

    def _initialize_database(
        self,
    ) -> None:
        """创建项目表并迁移已有 documents 表。

        第八天创建的数据库没有 project_id。
        这里先检查已有字段，只有缺少时才执行
        ALTER TABLE，因此每次启动都可以安全调用。
        """
        with self._connect() as connection:
            # documents 的外键指向 projects，
            # 所以先确保 projects 表存在。
            connection.executescript(CREATE_PROJECTS_SQL)

            connection.executescript(CREATE_DOCUMENTS_SQL)

            # PRAGMA table_info 返回表中的字段信息。
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }

            # CREATE TABLE IF NOT EXISTS 不会修改旧表，
            # 因此第八天数据库需要执行一次 ALTER。
            if "project_id" not in columns:
                connection.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN project_id TEXT
                        REFERENCES projects(project_id)
                    """
                )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_documents_project_id
                ON documents(project_id)
                """
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row | None) -> DocumentRecord | None:
        """将 SQLite Row 转为有类型提示的业务对象。"""
        if row is None:
            return None
        # 把字典的每个键值对变成构造函数的关键字参数
        return DocumentRecord(**dict(row))

    @staticmethod
    def _optional_text(value: str | None) -> str | None:
        """把空字符串统一转换为数据库中的 NULL。"""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @classmethod
    def _normalize_metadata(cls, metadata: DocumentMetadata) -> DocumentMetadata:
        """清理并校验用户填写的文档信息。"""
        title = metadata.title.strip()
        document_type = metadata.document_type.strip()
        if not title:
            raise ValueError("文档标题不能为空")
        if not document_type:
            raise ValueError("文档类型不能为空")

        report_date = cls._optional_text(metadata.report_date)
        if report_date is not None:
            try:  # 验证报告日期格式，返回 date(2026, 9, 28)
                date.fromisoformat(report_date)
            except ValueError as error:
                # 如果格式错误则抛出异常，date.fromisoformat("2026/09/28")
                # ❌ ValueError
                raise ValueError("报告日期必须使用 YYYY-MM-DD") from error

        ticker = cls._optional_text(metadata.ticker)
        return DocumentMetadata(
            title=title,
            document_type=document_type,
            company=cls._optional_text(metadata.company),
            ticker=ticker.upper() if ticker else None,
            industry=cls._optional_text(metadata.industry),
            report_date=report_date,
        )

    def get_by_id(self, document_id: str) -> DocumentRecord | None:
        """按业务 ID 查询文档。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self._row_to_record(row)

    def get_by_hash(self, document_hash: str) -> DocumentRecord | None:
        """按文件内容哈希查询，用于重复上传判断。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_hash = ?",
                (document_hash,),
            ).fetchone()
        return self._row_to_record(row)

    def get_by_saved_name(self, saved_name: str) -> DocumentRecord | None:
        """按磁盘和 Milvus 共用的文件名查询。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE saved_name = ?",
                (saved_name,),
            ).fetchone()
        return self._row_to_record(row)

    def list_documents(
        self,
        status: str | None = None,
        project_id: str | None = None,
        industry: str | None = None,
        company: str | None = None,
        report_date_from: str | None = None,
        report_date_to: str | None = None,
    ) -> list[DocumentRecord]:
        """按照多个可选条件组合筛选文档。

        Args:
            status:
                文档处理状态。
            project_id:
                文档所属研究项目。
                None 表示不限制项目。
            industry:
                精确匹配行业。
            company:
                精确匹配公司或机构。
            report_date_from:
                报告日期下限，格式为 YYYY-MM-DD。
            report_date_to:
                报告日期上限，格式为 YYYY-MM-DD。

        Returns:
            按上传时间倒序排列的文档记录。
        """
        conditions: list[str] = []
        parameters: list[str] = []

        if status is not None:
            if status not in VALID_DOCUMENT_STATUSES:
                raise ValueError(f"不支持的文档状态：{status}")

            conditions.append("status = ?")
            parameters.append(status)

        if project_id is not None:
            cleaned_project_id = project_id.strip()

            if not cleaned_project_id:
                raise ValueError("project_id 不能为空字符串")

            conditions.append("project_id = ?")
            parameters.append(cleaned_project_id)

        normalized_industry = self._optional_text(industry)

        if normalized_industry is not None:
            conditions.append("industry = ?")
            parameters.append(normalized_industry)

        normalized_company = self._optional_text(company)

        if normalized_company is not None:
            conditions.append("company = ?")
            parameters.append(normalized_company)

        normalized_date_from = self._optional_text(report_date_from)

        normalized_date_to = self._optional_text(report_date_to)

        # SQLite 使用 ISO 日期字符串时，
        # YYYY-MM-DD 的字符串顺序与日期顺序一致。
        if normalized_date_from is not None:
            try:
                # 把字符串解析为日期对象
                date.fromisoformat(normalized_date_from)
            except ValueError as error:
                raise ValueError("开始日期必须使用 YYYY-MM-DD") from error

            conditions.append("report_date >= ?")
            parameters.append(normalized_date_from)

        if normalized_date_to is not None:
            try:
                date.fromisoformat(normalized_date_to)
            except ValueError as error:
                raise ValueError("结束日期必须使用 YYYY-MM-DD") from error

            conditions.append("report_date <= ?")
            parameters.append(normalized_date_to)

        if (
            normalized_date_from is not None
            and normalized_date_to is not None
            and normalized_date_from > normalized_date_to
        ):
            raise ValueError("开始日期不能晚于结束日期")

        sql = "SELECT * FROM documents"

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY uploaded_at DESC, saved_name ASC"

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        return [DocumentRecord(**dict(row)) for row in rows]

    def get_project_by_id(
        self,
        project_id: str,
    ) -> ProjectRecord | None:
        """根据项目 ID 查询研究项目。"""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM projects
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()

        if row is None:
            return None

        return ProjectRecord(**dict(row))

    def create_project(
        self,
        name: str,
        description: str | None = None,
    ) -> ProjectRecord:
        """创建一个研究项目。

        项目名称不区分英文大小写，不能重复。
        """
        cleaned_name = name.strip()

        if not cleaned_name:
            raise ValueError("项目名称不能为空")

        cleaned_description = self._optional_text(description)

        # UUID 与名称无关，因此以后修改项目名称时，
        # 文档关联不需要变化。
        project_id = f"project_{uuid4().hex[:16]}"

        created_at = datetime.now(UTC).isoformat(timespec="seconds")

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO projects (
                        project_id,
                        name,
                        description,
                        created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        cleaned_name,
                        cleaned_description,
                        created_at,
                    ),
                )

        except sqlite3.IntegrityError as error:
            raise ValueError(f"项目名称已经存在：{cleaned_name}") from error

        return ProjectRecord(
            project_id=project_id,
            name=cleaned_name,
            description=cleaned_description,
            created_at=created_at,
        )

    def list_projects(
        self,
    ) -> list[ProjectRecord]:
        """按照创建时间列出所有研究项目。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM projects
                ORDER BY created_at, name
                """
            ).fetchall()

        return [ProjectRecord(**dict(row)) for row in rows]

    def assign_document(
        self,
        document_id: str,
        project_id: str | None,
    ) -> None:
        """将文档分配给项目。

        Args:
            document_id:
                等待修改的文档 ID。
            project_id:
                目标项目 ID。
                传入 None 表示移出当前项目。
        """
        with self._connect() as connection:
            if project_id is not None:
                project = connection.execute(
                    """
                    SELECT project_id
                    FROM projects
                    WHERE project_id = ?
                    """,
                    (project_id,),
                ).fetchone()

                if project is None:
                    raise KeyError(f"找不到研究项目：{project_id}")

            cursor = connection.execute(
                """
                UPDATE documents
                SET project_id = ?
                WHERE document_id = ?
                """,
                (
                    project_id,
                    document_id,
                ),
            )

            if cursor.rowcount != 1:
                raise KeyError(f"找不到文档：{document_id}")

    def delete_document(
        self,
        document_id: str,
    ) -> DocumentRecord:
        """删除SQLite中的文档登记记录。

        这个方法只管理SQLite，不负责删除Milvus证据
        和data/raw中的PDF。跨存储清理会交给上层服务协调。

        Returns:
            删除前的完整文档记录。
        """
        cleaned_document_id = document_id.strip()

        if not cleaned_document_id:
            raise ValueError("document_id不能为空")

        with self._connect() as connection:
            # 先读取完整记录，后续删除本地文件时
            # 仍然需要saved_name等信息。
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE document_id = ?
                """,
                (cleaned_document_id,),
            ).fetchone()

            if row is None:
                raise KeyError(f"找不到文档：{cleaned_document_id}")

            connection.execute(
                """
                DELETE FROM documents
                WHERE document_id = ?
                """,
                (cleaned_document_id,),
            )

        deleted_record = self._row_to_record(row)

        # 前面已经检查过row不为空，
        # 这个判断主要帮助类型检查器确认返回类型。
        if deleted_record is None:
            raise RuntimeError("删除文档后无法恢复原记录")

        return deleted_record

    def register_document(
        self,
        document_hash: str,
        original_name: str,
        saved_name: str,
        metadata: DocumentMetadata,
        total_pages: int = 0,
    ) -> tuple[DocumentRecord, bool]:
        """登记新文档；相同哈希已存在时直接返回旧记录。"""
        existing = self.get_by_hash(document_hash)
        if existing is not None:
            return existing, False
        if len(document_hash) != 64:
            raise ValueError("document_hash 必须是完整 SHA-256")
        if total_pages < 0:
            raise ValueError("total_pages 不能小于 0")

        normalized = self._normalize_metadata(metadata)
        document_id = f"doc_{document_hash[:20]}"
        uploaded_at = datetime.now(UTC).isoformat(timespec="seconds")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, document_hash, original_name, saved_name,
                    title, company, ticker, industry, document_type,
                    report_date, uploaded_at, total_pages, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    document_hash,
                    original_name,
                    saved_name,
                    normalized.title,
                    normalized.company,
                    normalized.ticker,
                    normalized.industry,
                    normalized.document_type,
                    normalized.report_date,
                    uploaded_at,
                    total_pages,
                    STATUS_SAVED,
                ),
            )

        record = self.get_by_id(document_id)
        if record is None:
            raise RuntimeError("文档登记完成后无法读取记录")
        return record, True

    def mark_processing(self, document_id: str) -> None:
        """将文档标记为处理中，并清除旧错误。"""
        self._update_status(document_id, STATUS_PROCESSING, None)

    def mark_failed(self, document_id: str, error_message: str) -> None:
        """保存失败状态和便于排查的错误摘要。"""
        cleaned_error = error_message.strip()[:1000] or "未知错误"
        self._update_status(document_id, STATUS_FAILED, cleaned_error)

    def mark_indexed(
        self,
        document_id: str,
        total_pages: int,
        processed_pages: int,
        text_chunk_count: int,
        image_chunk_count: int = 0,
    ) -> None:
        """保存成功入库后的页数和证据统计。"""
        counts = (total_pages, processed_pages, text_chunk_count, image_chunk_count)
        if any(value < 0 for value in counts):
            raise ValueError("页数和证据数量不能小于 0")
        if processed_pages > total_pages:
            raise ValueError("处理页数不能超过 PDF 总页数")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET total_pages = ?, processed_pages = ?,
                    text_chunk_count = ?, image_chunk_count = ?,
                    status = ?, error_message = NULL
                WHERE document_id = ?
                """,
                (
                    total_pages,
                    processed_pages,
                    text_chunk_count,
                    image_chunk_count,
                    STATUS_INDEXED,
                    document_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"找不到文档：{document_id}")

    def update_image_chunk_count(
        self,
        document_id: str,
        image_chunk_count: int,
    ) -> None:
        """更新一份文档实际拥有的图像证据数量。

        这个方法不会修改文档状态和文本证据数。
        图表页处理失败时，原有文本RAG仍然可用。
        """
        cleaned_document_id = document_id.strip()

        if not cleaned_document_id:
            raise ValueError("document_id不能为空")

        if image_chunk_count < 0:
            raise ValueError("图像证据数量不能小于0")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET image_chunk_count = ?
                WHERE document_id = ?
                """,
                (
                    image_chunk_count,
                    cleaned_document_id,
                ),
            )

            if cursor.rowcount != 1:
                raise KeyError(f"找不到文档：{cleaned_document_id}")

    def _update_status(
        self,
        document_id: str,
        status: str,
        error_message: str | None,
    ) -> None:
        """集中执行只修改状态的 SQL。"""
        if status not in VALID_DOCUMENT_STATUSES:
            raise ValueError(f"不支持的文档状态：{status}")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE documents SET status = ?, error_message = ? WHERE document_id = ?",
                (status, error_message, document_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"找不到文档：{document_id}")
