"""使用 SQLite 管理文档元数据和处理状态。"""
from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Final

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
    error_message TEXT
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
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

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

    def _initialize_database(self) -> None:
        """重复执行也安全地创建表与索引。"""
        with self._connect() as connection:
            # executescript 会自动提交事务，从而完成建表流程
            connection.executescript(CREATE_DOCUMENTS_SQL)

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
            try: # 验证报告日期格式，返回 date(2026, 9, 28)
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

    def list_documents(self, status: str | None = None) -> list[DocumentRecord]:
        """按上传时间倒序列出文档，可限定处理状态。"""
        parameters: tuple[str, ...] = ()
        sql = "SELECT * FROM documents"
        if status is not None:
            if status not in VALID_DOCUMENT_STATUSES:
                raise ValueError(f"不支持的文档状态：{status}")
            sql += " WHERE status = ?"
            parameters = (status,)
        sql += " ORDER BY uploaded_at DESC, saved_name ASC"

        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

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
        uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

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
                    document_id, document_hash, original_name, saved_name,
                    normalized.title, normalized.company, normalized.ticker,
                    normalized.industry, normalized.document_type,
                    normalized.report_date, uploaded_at, total_pages,
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
                    total_pages, processed_pages, text_chunk_count,
                    image_chunk_count, STATUS_INDEXED, document_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"找不到文档：{document_id}")

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