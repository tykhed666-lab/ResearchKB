"""查询和解析 SEC EDGAR 官方披露数据。"""

from dataclasses import dataclass
import os
import re
from time import monotonic, sleep
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from research_kb.settings import PROJECT_ROOT

# SEC 的 User-Agent 从项目根目录的 .env 读取。
ENV_PATH = PROJECT_ROOT / ".env"
# SEC 提供的股票代码、CIK、公司名称映射。
SEC_TICKER_MAPPING_URL = (
    "https://www.sec.gov/files/"
    "company_tickers.json"
)

# 公司近期申报记录接口。
# cik 必须是包含前导零的十位字符串。
SEC_SUBMISSIONS_URL_TEMPLATE = (
    "https://data.sec.gov/submissions/"
    "CIK{cik}.json"
)

# SEC 归档文件的根地址。
SEC_ARCHIVES_BASE_URL = (
    "https://www.sec.gov/Archives/edgar/data"
)

# 第十一天只处理年报和季报。
SUPPORTED_FORMS = frozenset(
    {
        "10-K",
        "10-Q",
    }
)

# accession number 的标准格式：
# 10位数字-2位年份-6位流水号。
ACCESSION_PATTERN = re.compile(
    r"^\d{10}-\d{2}-\d{6}$"
)


@dataclass(
    frozen=True,
    slots=True,
)
class SecCompany:
    """表示 SEC 中的一家上市公司。

    Attributes:
        ticker:
            股票代码，例如 NVDA。
        cik:
            SEC 为公司分配的十位 CIK。
        name:
            SEC 使用的公司名称。
    """

    ticker: str
    cik: str
    name: str


@dataclass(
    frozen=True,
    slots=True,
)
class SecFiling:
    """表示一份 SEC 公司披露。

    Attributes:
        source_id:
            本项目为披露生成的稳定 ID。
        company:
            当前披露所属的公司。
        form_type:
            申报类型，例如 10-K 或 10-Q。
        accession_number:
            SEC 为本次申报分配的唯一编号。
        filing_date:
            文件提交到 SEC 的日期。
        report_date:
            这份报告对应的报告期日期。
        primary_document:
            本次申报的主 HTML 文件名。
        document_url:
            主 HTML 文件的官方地址。
        index_url:
            本次申报的文件目录页面。
    """

    source_id: str
    company: SecCompany
    form_type: str
    accession_number: str
    filing_date: str
    report_date: str | None
    primary_document: str
    document_url: str
    index_url: str


def normalize_ticker(
    ticker: str,
) -> str:
    """清理用户输入的股票代码。

    示例：
        " nvda " -> "NVDA"

    Raises:
        ValueError:
            股票代码为空。
    """
    normalized = ticker.strip().upper()

    if not normalized:
        raise ValueError(
            "股票代码不能为空"
        )

    return normalized


def format_cik(
    cik: str | int,
) -> str:
    """将 CIK 格式化为十位字符串。

    SEC Submissions API 要求 CIK 包含前导零。

    示例：
        1045810 -> "0001045810"
    """
    cleaned = str(cik).strip()

    if (
        not cleaned
        or not cleaned.isdigit()
    ):
        raise ValueError(
            "CIK 必须是数字"
        )

    if len(cleaned) > 10:
        raise ValueError(
            "CIK 不能超过十位"
        )

    if int(cleaned) <= 0:
        raise ValueError(
            "CIK 必须大于 0"
        )

    return cleaned.zfill(10)


def normalize_accession_number(
    accession_number: str,
) -> str:
    """校验并返回标准 accession number。

    示例：
        0001193125-15-118890
    """
    cleaned = accession_number.strip()

    if not ACCESSION_PATTERN.fullmatch(
        cleaned
    ):
        raise ValueError(
            "accession number 格式不正确"
        )

    return cleaned


def validate_primary_document(
    primary_document: str,
) -> str:
    """校验 SEC 返回的主文档文件名。

    主文档应当只是文件名，不能包含目录。
    """
    cleaned = primary_document.strip()

    if not cleaned:
        raise ValueError(
            "SEC 主文档文件名不能为空"
        )

    if (
        "/" in cleaned
        or "\\" in cleaned
    ):
        raise ValueError(
            "SEC 主文档文件名不能包含目录"
        )

    return cleaned


def build_filing_url(
    cik: str | int,
    accession_number: str,
    primary_document: str,
) -> str:
    """构造 SEC 披露主文档的官方 URL。

    SEC 归档目录中的规则：

    1. CIK 不包含前导零；
    2. accession number 删除连字符；
    3. 最后一部分是主文档文件名。
    """
    formatted_cik = format_cik(cik)

    normalized_accession = (
        normalize_accession_number(
            accession_number
        )
    )

    cleaned_document = (
        validate_primary_document(
            primary_document
        )
    )

    # int() 会去掉 CIK 前面的零。
    cik_directory = str(
        int(formatted_cik)
    )

    accession_directory = (
        normalized_accession.replace(
            "-",
            "",
        )
    )

    # 对文件名进行 URL 编码，
    # 同时保留常见的文件名符号。
    encoded_document = quote(
        cleaned_document,
        safe="._~-",
    )

    return (
        f"{SEC_ARCHIVES_BASE_URL}/"
        f"{cik_directory}/"
        f"{accession_directory}/"
        f"{encoded_document}"
    )


def build_filing_index_url(
    cik: str | int,
    accession_number: str,
) -> str:
    """构造一份申报对应的 SEC 文件目录页面。"""
    formatted_cik = format_cik(cik)

    normalized_accession = (
        normalize_accession_number(
            accession_number
        )
    )

    cik_directory = str(
        int(formatted_cik)
    )

    accession_directory = (
        normalized_accession.replace(
            "-",
            "",
        )
    )

    return (
        f"{SEC_ARCHIVES_BASE_URL}/"
        f"{cik_directory}/"
        f"{accession_directory}/"
        f"{normalized_accession}"
        "-index.html"
    )


def parse_company_mapping(
    payload: dict[str, Any],
) -> dict[str, SecCompany]:
    """解析 SEC 的 Ticker-CIK 映射 JSON。

    Args:
        payload:
            SEC 接口返回并解析后的完整 JSON 字典。

    Returns:
        以大写 Ticker 为键，
        SecCompany 为值的字典。
    """
    companies: dict[
        str,
        SecCompany,
    ] = {}

    for raw_record in payload.values():
        if not isinstance(
            raw_record,
            dict,
        ):
            continue

        raw_ticker = raw_record.get(
            "ticker"
        )
        raw_cik = raw_record.get(
            "cik_str"
        )
        raw_name = raw_record.get(
            "title"
        )

        if (
            raw_ticker is None
            or raw_cik is None
            or raw_name is None
        ):
            continue

        try:
            ticker = normalize_ticker(
                str(raw_ticker)
            )
            cik = format_cik(raw_cik)

        except ValueError:
            # 一条异常记录不应导致
            # 整个公司映射解析失败。
            continue

        company_name = str(
            raw_name
        ).strip()

        if not company_name:
            continue

        companies[ticker] = SecCompany(
            ticker=ticker,
            cik=cik,
            name=company_name,
        )

    return companies


def get_company_from_mapping(
    ticker: str,
    companies: dict[
        str,
        SecCompany,
    ],
) -> SecCompany:
    """根据股票代码取得公司记录。"""
    normalized_ticker = (
        normalize_ticker(ticker)
    )

    company = companies.get(
        normalized_ticker
    )

    if company is None:
        raise LookupError(
            "SEC 公司列表中找不到股票代码："
            f"{normalized_ticker}"
        )

    return company


def _get_column_value(
    columns: dict[str, Any],
    column_name: str,
    index: int,
) -> Any | None:
    """安全读取 SEC 并列数组中的一个值。

    SEC 的 recent filings 使用 columnar JSON：

    form[0]、filingDate[0]、reportDate[0]
    共同组成第一条申报。

    某个数组缺少当前位置时返回 None，
    防止异常记录造成整个查询失败。
    """
    values = columns.get(
        column_name
    )

    if not isinstance(values, list):
        return None

    if index >= len(values):
        return None

    return values[index]


def parse_recent_filings(
    company: SecCompany,
    payload: dict[str, Any],
    limit: int = 10,
) -> list[SecFiling]:
    """解析公司的近期 10-K 和 10-Q。

    Args:
        company:
            当前查询的公司。
        payload:
            Submissions API 返回的 JSON。
        limit:
            最多返回多少份披露。

    Returns:
        按 SEC 原始顺序排列的披露列表。
    """
    if limit <= 0:
        raise ValueError(
            "披露数量限制必须大于 0"
        )

    filings = payload.get(
        "filings"
    )

    if not isinstance(filings, dict):
        return []

    recent = filings.get(
        "recent"
    )

    if not isinstance(recent, dict):
        return []

    forms = recent.get(
        "form"
    )

    if not isinstance(forms, list):
        return []

    results: list[SecFiling] = []

    for index, raw_form in enumerate(
        forms
    ):
        form_type = str(
            raw_form
        ).strip().upper()

        # 8-K、Form 4 等其他文件
        # 暂时不属于第十一天的范围。
        if form_type not in SUPPORTED_FORMS:
            continue

        accession_number = (
            _get_column_value(
                recent,
                "accessionNumber",
                index,
            )
        )

        filing_date = (
            _get_column_value(
                recent,
                "filingDate",
                index,
            )
        )

        report_date = (
            _get_column_value(
                recent,
                "reportDate",
                index,
            )
        )

        primary_document = (
            _get_column_value(
                recent,
                "primaryDocument",
                index,
            )
        )

        # 构造链接所需的字段缺失时，
        # 当前记录无法正常使用，因此跳过。
        if (
            not accession_number
            or not filing_date
            or not primary_document
        ):
            continue

        try:
            normalized_accession = (
                normalize_accession_number(
                    str(
                        accession_number
                    )
                )
            )

            cleaned_document = (
                validate_primary_document(
                    str(
                        primary_document
                    )
                )
            )

        except ValueError:
            continue

        cleaned_report_date = (
            str(report_date).strip()
            if report_date
            else None
        )

        accession_without_dashes = (
            normalized_accession.replace(
                "-",
                "",
            )
        )

        results.append(
            SecFiling(
                source_id=(
                    f"sec_{company.cik}_"
                    f"{accession_without_dashes}"
                ),
                company=company,
                form_type=form_type,
                accession_number=(
                    normalized_accession
                ),
                filing_date=str(
                    filing_date
                ).strip(),
                report_date=(
                    cleaned_report_date
                    or None
                ),
                primary_document=(
                    cleaned_document
                ),
                document_url=(
                    build_filing_url(
                        cik=company.cik,
                        accession_number=(
                            normalized_accession
                        ),
                        primary_document=(
                            cleaned_document
                        ),
                    )
                ),
                index_url=(
                    build_filing_index_url(
                        cik=company.cik,
                        accession_number=(
                            normalized_accession
                        ),
                    )
                ),
            )
        )

        if len(results) >= limit:
            break

    return results


class SecEdgarError(RuntimeError):
    """SEC EDGAR 访问过程中的基础异常。"""


class SecConfigurationError(
    SecEdgarError
):
    """SEC 客户端配置不正确。"""


class SecRequestError(
    SecEdgarError
):
    """SEC 网络请求或响应内容不正确。"""


class SecRateLimitError(
    SecRequestError
):
    """SEC 接口返回请求频率限制。"""


class SecEdgarClient:
    """访问 SEC EDGAR 官方数据接口。

    这个类负责：

    1. 读取和校验 User-Agent；
    2. 控制请求频率；
    3. 处理超时、限流和服务端错误；
    4. 下载 Ticker-CIK 映射；
    5. 查询公司的近期 10-K 和 10-Q。
    """

    def __init__(
        self,
        user_agent: str | None = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        requests_per_second: float = 5.0,
        http_client: (
            httpx.Client
            | None
        ) = None,
    ) -> None:
        """创建 SEC EDGAR 客户端。

        Args:
            user_agent:
                应用名称和联系邮箱。
                未传入时从 SEC_USER_AGENT 读取。
            timeout_seconds:
                单次网络请求的超时时间。
            max_retries:
                请求失败后的最大重试次数。
            requests_per_second:
                每秒允许发送的请求数。
                项目默认使用 5，低于 SEC 的上限。
            http_client:
                可选的 httpx.Client。
                自动化测试可以传入模拟客户端。
        """
        load_dotenv(ENV_PATH)

        configured_user_agent = (
            user_agent
            or os.getenv(
                "SEC_USER_AGENT"
            )
            or ""
        ).strip()

        if not configured_user_agent:
            raise SecConfigurationError(
                "缺少环境变量："
                "SEC_USER_AGENT"
            )

        if "@" not in configured_user_agent:
            raise SecConfigurationError(
                "SEC_USER_AGENT 应包含"
                "应用名称和联系邮箱"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds 必须大于 0"
            )

        if max_retries < 0:
            raise ValueError(
                "max_retries 不能小于 0"
            )

        if requests_per_second <= 0:
            raise ValueError(
                "requests_per_second 必须大于 0"
            )

        self.user_agent = (
            configured_user_agent
        )
        self.timeout_seconds = (
            timeout_seconds
        )
        self.max_retries = max_retries

        # 5次/秒对应最小间隔0.2秒。
        self.minimum_interval = (
            1.0
            / requests_per_second
        )

        self._last_request_time: (
            float
            | None
        ) = None

        # 缓存公司映射。
        # 同一个客户端查询多个股票时，
        # 不需要重复下载整个映射文件。
        self._company_mapping: (
            dict[str, SecCompany]
            | None
        ) = None

        # 外部传入的客户端通常由测试代码管理；
        # 内部创建的客户端由本类负责关闭。
        self._owns_http_client = (
            http_client is None
        )

        self._http_client = (
            http_client
            or httpx.Client(
                timeout=timeout_seconds
            )
        )

        self._request_headers = {
            "User-Agent": self.user_agent,
            "Accept": (
                "application/json, "
                "text/plain;q=0.9, "
                "*/*;q=0.8"
            ),
            "Accept-Encoding": (
                "gzip, deflate"
            ),
        }

    @property
    def is_configured(self) -> bool:
        """客户端构造成功即表示配置有效。"""
        return True

    def _wait_for_rate_limit(
        self,
    ) -> None:
        """在请求之间保持最小时间间隔。"""
        if self._last_request_time is None:
            return

        elapsed = (
            monotonic()
            - self._last_request_time
        )

        remaining = (
            self.minimum_interval
            - elapsed
        )

        if remaining > 0:
            sleep(remaining)

    def _calculate_retry_delay(
        self,
        attempt: int,
        response: (
            httpx.Response
            | None
        ) = None,
    ) -> float:
        """计算下一次重试前等待的时间。"""
        default_delay = float(
            attempt + 1
        )

        if response is None:
            return default_delay

        retry_after = (
            response.headers.get(
                "Retry-After"
            )
        )

        if retry_after is None:
            return default_delay

        try:
            server_delay = float(
                retry_after
            )
        except ValueError:
            return default_delay

        # 页面请求不应因为异常响应头
        # 无限制等待。
        return min(
            max(
                server_delay,
                default_delay,
            ),
            5.0,
        )

    def _get_json(
        self,
        url: str,
    ) -> dict[str, Any]:
        """请求一个 SEC JSON 地址。

        对超时、网络错误、429 和 5xx
        执行有限次数重试。
        """
        last_error: Exception | None = None

        for attempt in range(
            self.max_retries + 1
        ):
            self._wait_for_rate_limit()

            # 记录请求开始时间，
            # 下一次请求会据此控制频率。
            self._last_request_time = (
                monotonic()
            )

            try:
                response = (
                    self._http_client.get(
                        url,
                        headers=(
                            self._request_headers
                        ),
                    )
                )

            except httpx.TimeoutException as error:
                last_error = error

                if attempt >= self.max_retries:
                    raise SecRequestError(
                        "连接 SEC 超时，请稍后重试"
                    ) from error

                sleep(
                    self._calculate_retry_delay(
                        attempt
                    )
                )
                continue

            except httpx.RequestError as error:
                last_error = error

                if attempt >= self.max_retries:
                    raise SecRequestError(
                        "无法连接 SEC，请检查网络"
                    ) from error

                sleep(
                    self._calculate_retry_delay(
                        attempt
                    )
                )
                continue

            if response.status_code == 429:
                if attempt >= self.max_retries:
                    raise SecRateLimitError(
                        "SEC 请求频率受到限制，"
                        "请稍后重试"
                    )

                sleep(
                    self._calculate_retry_delay(
                        attempt,
                        response,
                    )
                )
                continue

            if (
                500
                <= response.status_code
                < 600
            ):
                if attempt >= self.max_retries:
                    raise SecRequestError(
                        "SEC 服务暂时不可用，"
                        f"HTTP {response.status_code}"
                    )

                sleep(
                    self._calculate_retry_delay(
                        attempt,
                        response,
                    )
                )
                continue

            if response.status_code >= 400:
                raise SecRequestError(
                    "SEC 请求失败，"
                    f"HTTP {response.status_code}"
                )

            try:
                payload = response.json()

            except ValueError as error:
                raise SecRequestError(
                    "SEC 返回的内容不是有效 JSON"
                ) from error

            if not isinstance(
                payload,
                dict,
            ):
                raise SecRequestError(
                    "SEC 返回的 JSON 结构不正确"
                )

            return payload

        # 正常情况下不会运行到这里，
        # 保留此分支让返回路径完整。
        raise SecRequestError(
            "SEC 请求失败"
        ) from last_error

    def download_document(
        self,
        url: str,
        max_bytes: int = 20 * 1024 * 1024,
    ) -> bytes:
        """下载一份 SEC 主 HTML 文档。

        复用 JSON 请求相同的限流、重试和错误转换规则，
        同时限制响应大小，避免意外把超大附件读入内存。
        """
        cleaned_url = url.strip()

        if not cleaned_url.startswith(
            "https://www.sec.gov/Archives/"
        ):
            raise ValueError(
                "只允许下载 SEC Archives 官方文档"
            )

        if max_bytes <= 0:
            raise ValueError(
                "max_bytes 必须大于 0"
            )

        last_error: Exception | None = None

        for attempt in range(
            self.max_retries + 1
        ):
            self._wait_for_rate_limit()
            self._last_request_time = monotonic()

            try:
                response = self._http_client.get(
                    cleaned_url,
                    headers=self._request_headers,
                )
            except httpx.TimeoutException as error:
                last_error = error
                if attempt >= self.max_retries:
                    raise SecRequestError(
                        "下载 SEC 原文超时，请稍后重试"
                    ) from error
                sleep(self._calculate_retry_delay(attempt))
                continue
            except httpx.RequestError as error:
                last_error = error
                if attempt >= self.max_retries:
                    raise SecRequestError(
                        "无法下载 SEC 原文，请检查网络"
                    ) from error
                sleep(self._calculate_retry_delay(attempt))
                continue

            if response.status_code == 429:
                if attempt >= self.max_retries:
                    raise SecRateLimitError(
                        "SEC 请求频率受到限制，请稍后重试"
                    )
                sleep(
                    self._calculate_retry_delay(
                        attempt,
                        response,
                    )
                )
                continue

            if 500 <= response.status_code < 600:
                if attempt >= self.max_retries:
                    raise SecRequestError(
                        "SEC 服务暂时不可用，"
                        f"HTTP {response.status_code}"
                    )
                sleep(
                    self._calculate_retry_delay(
                        attempt,
                        response,
                    )
                )
                continue

            if response.status_code >= 400:
                raise SecRequestError(
                    "SEC 原文下载失败，"
                    f"HTTP {response.status_code}"
                )

            content = response.content
            if len(content) > max_bytes:
                raise SecRequestError(
                    "SEC 原文超过 20 MB 限制"
                )

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()
            if (
                "html" not in content_type
                and b"<html" not in content[:2048].lower()
            ):
                raise SecRequestError(
                    "SEC 返回的主文档不是 HTML"
                )

            return content

        raise SecRequestError(
            "SEC 原文下载失败"
        ) from last_error

    def load_company_mapping(
        self,
        force_refresh: bool = False,
    ) -> dict[str, SecCompany]:
        """下载并缓存 Ticker-CIK 公司映射。

        Args:
            force_refresh:
                True 表示忽略内存缓存，
                重新向 SEC 请求。
        """
        if (
            self._company_mapping
            is not None
            and not force_refresh
        ):
            return self._company_mapping

        payload = self._get_json(
            SEC_TICKER_MAPPING_URL
        )

        companies = (
            parse_company_mapping(
                payload
            )
        )

        if not companies:
            raise SecRequestError(
                "SEC 公司映射为空"
            )

        self._company_mapping = companies

        return companies

    def resolve_company(
        self,
        ticker: str,
    ) -> SecCompany:
        """将股票代码解析为 SEC 公司记录。"""
        companies = (
            self.load_company_mapping()
        )

        return get_company_from_mapping(
            ticker=ticker,
            companies=companies,
        )

    def get_recent_filings(
        self,
        ticker: str,
        limit: int = 10,
    ) -> list[SecFiling]:
        """查询一家公司的近期 10-K 和 10-Q。"""
        if limit <= 0:
            raise ValueError(
                "披露数量限制必须大于 0"
            )

        company = self.resolve_company(
            ticker
        )

        submissions_url = (
            SEC_SUBMISSIONS_URL_TEMPLATE.format(
                cik=company.cik
            )
        )

        payload = self._get_json(
            submissions_url
        )

        return parse_recent_filings(
            company=company,
            payload=payload,
            limit=limit,
        )

    def close(self) -> None:
        """关闭本类自行创建的 HTTP 客户端。"""
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(
        self,
    ) -> "SecEdgarClient":
        """支持 with SecEdgarClient() 写法。"""
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        """离开 with 代码块时关闭连接。"""
        self.close()
