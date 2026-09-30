"""测试 SEC EDGAR 数据解析和网络客户端。"""

import httpx
import pytest

from research_kb.sec_edgar import (
    SEC_SUBMISSIONS_URL_TEMPLATE,
    SEC_TICKER_MAPPING_URL,
    SecCompany,
    SecConfigurationError,
    SecEdgarClient,
    SecRateLimitError,
    SecRequestError,
    build_filing_index_url,
    build_filing_url,
    format_cik,
    normalize_ticker,
    parse_company_mapping,
    parse_recent_filings,
)


def build_company_mapping_payload(
) -> dict:
    """构造测试使用的公司映射数据。"""
    return {
        "0": {
            "cik_str": 1045810,
            "ticker": "NVDA",
            "title": "NVIDIA CORP",
        },
        "1": {
            "cik_str": 27419,
            "ticker": "TGT",
            "title": "TARGET CORP",
        },
    }


def build_submissions_payload(
) -> dict:
    """构造测试使用的近期申报数据。"""
    return {
        "filings": {
            "recent": {
                "form": [
                    "8-K",
                    "10-K",
                    "10-Q",
                ],
                "accessionNumber": [
                    "0001045810-26-000001",
                    "0001045810-26-000002",
                    "0001045810-25-000003",
                ],
                "filingDate": [
                    "2026-01-01",
                    "2026-02-25",
                    "2025-11-19",
                ],
                "reportDate": [
                    "",
                    "2026-01-25",
                    "2025-10-26",
                ],
                "primaryDocument": [
                    "eight-k.htm",
                    "annual.htm",
                    "quarterly.htm",
                ],
            }
        }
    }


def test_normalize_ticker_and_cik(
) -> None:
    """Ticker应大写，CIK应补足十位。"""
    assert normalize_ticker(
        " nvda "
    ) == "NVDA"

    assert format_cik(
        1045810
    ) == "0001045810"

    with pytest.raises(
        ValueError,
        match="股票代码不能为空",
    ):
        normalize_ticker("   ")

    with pytest.raises(
        ValueError,
        match="CIK 必须是数字",
    ):
        format_cik("abc")


def test_parse_company_mapping(
) -> None:
    """公司映射应转换为SecCompany对象。"""
    companies = parse_company_mapping(
        build_company_mapping_payload()
    )

    assert companies["NVDA"] == (
        SecCompany(
            ticker="NVDA",
            cik="0001045810",
            name="NVIDIA CORP",
        )
    )

    assert companies["TGT"].cik == (
        "0000027419"
    )


def test_build_filing_urls(
) -> None:
    """应按SEC归档规则构造官方地址。"""
    document_url = build_filing_url(
        cik="0001045810",
        accession_number=(
            "0001045810-26-000021"
        ),
        primary_document=(
            "nvda-20260125.htm"
        ),
    )

    assert document_url == (
        "https://www.sec.gov/Archives/"
        "edgar/data/1045810/"
        "000104581026000021/"
        "nvda-20260125.htm"
    )

    index_url = (
        build_filing_index_url(
            cik="0001045810",
            accession_number=(
                "0001045810-26-000021"
            ),
        )
    )

    assert index_url == (
        "https://www.sec.gov/Archives/"
        "edgar/data/1045810/"
        "000104581026000021/"
        "0001045810-26-000021-index.html"
    )


def test_parse_only_10k_and_10q(
) -> None:
    """近期申报应过滤掉8-K等其他表单。"""
    company = SecCompany(
        ticker="NVDA",
        cik="0001045810",
        name="NVIDIA CORP",
    )

    filings = parse_recent_filings(
        company=company,
        payload=(
            build_submissions_payload()
        ),
        limit=10,
    )

    assert [
        filing.form_type
        for filing in filings
    ] == [
        "10-K",
        "10-Q",
    ]

    assert (
        filings[0].report_date
        == "2026-01-25"
    )

    assert (
        filings[0].company
        == company
    )


def test_incomplete_parallel_arrays_are_skipped(
) -> None:
    """字段数组长度不一致时跳过不完整记录。"""
    company = SecCompany(
        ticker="NVDA",
        cik="0001045810",
        name="NVIDIA CORP",
    )

    payload = {
        "filings": {
            "recent": {
                "form": [
                    "10-K",
                    "10-Q",
                ],
                "accessionNumber": [
                    "0001045810-26-000021",
                    "0001045810-26-000052",
                ],
                # 第二条记录缺少提交日期。
                "filingDate": [
                    "2026-02-25",
                ],
                "reportDate": [
                    "2026-01-25",
                    "2026-04-26",
                ],
                "primaryDocument": [
                    "annual.htm",
                    "quarterly.htm",
                ],
            }
        }
    }

    filings = parse_recent_filings(
        company=company,
        payload=payload,
    )

    assert len(filings) == 1
    assert (
        filings[0].form_type
        == "10-K"
    )


def test_client_uses_mapping_cache(
) -> None:
    """重复解析公司时不应重复下载映射。"""
    request_counts = {
        "mapping": 0,
        "submissions": 0,
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        """返回固定的模拟SEC响应。"""
        assert (
            request.headers["User-Agent"]
            == "ResearchKB tests@example.com"
        )

        if (
            str(request.url)
            == SEC_TICKER_MAPPING_URL
        ):
            request_counts[
                "mapping"
            ] += 1

            return httpx.Response(
                status_code=200,
                json=(
                    build_company_mapping_payload()
                ),
            )

        expected_url = (
            SEC_SUBMISSIONS_URL_TEMPLATE
            .format(
                cik="0001045810"
            )
        )

        if str(request.url) == expected_url:
            request_counts[
                "submissions"
            ] += 1

            return httpx.Response(
                status_code=200,
                json=(
                    build_submissions_payload()
                ),
            )

        return httpx.Response(
            status_code=404
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    client = SecEdgarClient(
        user_agent=(
            "ResearchKB tests@example.com"
        ),
        max_retries=0,
        requests_per_second=100000,
        http_client=http_client,
    )

    try:
        first_company = (
            client.resolve_company(
                "NVDA"
            )
        )

        second_company = (
            client.resolve_company(
                " nvda "
            )
        )

        filings = (
            client.get_recent_filings(
                "NVDA",
                limit=5,
            )
        )

    finally:
        http_client.close()

    assert first_company == second_company

    assert (
        request_counts["mapping"]
        == 1
    )

    assert (
        request_counts["submissions"]
        == 1
    )

    assert len(filings) == 2


def test_unknown_ticker_raises_lookup_error(
) -> None:
    """未知Ticker应提供明确错误。"""
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=(
                build_company_mapping_payload()
            ),
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    client = SecEdgarClient(
        user_agent=(
            "ResearchKB tests@example.com"
        ),
        max_retries=0,
        requests_per_second=100000,
        http_client=http_client,
    )

    try:
        with pytest.raises(
            LookupError,
            match="找不到股票代码",
        ):
            client.resolve_company(
                "UNKNOWN"
            )

    finally:
        http_client.close()


def test_rate_limit_error(
) -> None:
    """HTTP 429应转换成专门的限流错误。"""
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={
                "Retry-After": "1",
            },
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    client = SecEdgarClient(
        user_agent=(
            "ResearchKB tests@example.com"
        ),
        max_retries=0,
        requests_per_second=100000,
        http_client=http_client,
    )

    try:
        with pytest.raises(
            SecRateLimitError,
            match="请求频率",
        ):
            client.load_company_mapping()

    finally:
        http_client.close()


def test_network_timeout_is_readable(
) -> None:
    """网络超时应转换成业务错误。"""
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ReadTimeout(
            "timeout",
            request=request,
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(
            handler
        )
    )

    client = SecEdgarClient(
        user_agent=(
            "ResearchKB tests@example.com"
        ),
        max_retries=0,
        requests_per_second=100000,
        http_client=http_client,
    )

    try:
        with pytest.raises(
            SecRequestError,
            match="连接 SEC 超时",
        ):
            client.load_company_mapping()

    finally:
        http_client.close()


def test_user_agent_is_required(
) -> None:
    """空User-Agent应在发送请求前被拒绝。"""
    with pytest.raises(
        SecConfigurationError,
        match="SEC_USER_AGENT",
    ):
        SecEdgarClient(
            user_agent=" ",
        )