from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import MilvusClient
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def check_packages() -> dict[str, str]:
    """读取关键依赖的实际安装版本。"""
    names = [
        "langchain-core",
        "langchain-openai",
        "langchain-text-splitters",
        "pymilvus",
        "pymupdf",
        "pypdf",
        "streamlit",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def check_pdfs() -> dict[str, object]:
    """统计原始 PDF 数量和每份文档的页数。"""
    # 排序后输出顺序固定，方便比较不同机器或不同日期的检查结果。
    files = sorted(RAW_DIR.glob("*.pdf"))
    page_counts = {path.name: len(PdfReader(str(path)).pages) for path in files}
    return {
        "count": len(files),
        "total_pages": sum(page_counts.values()),
        "files": page_counts,
    }


def check_milvus(uri: str) -> dict[str, object]:
    """连接 Milvus，并通过列出 Collection 验证协议连接。"""
    client = MilvusClient(uri=uri)
    return {"uri": uri, "collections": client.list_collections()}


def main() -> int:
    """汇总 Python、数据、Milvus 和云模型配置检查结果。"""
    load_dotenv(ROOT / ".env")
    milvus_uri = os.getenv("MILVUS_URI", "http://192.168.88.161:19530")

    # 只输出配置是否齐全，不输出任何 API Key。
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "packages": check_packages(),
        "pdfs": check_pdfs(),
        "milvus": check_milvus(milvus_uri),
        "cloud_model_configured": bool(
            os.getenv("OPENAI_API_KEY")
            and os.getenv("OPENAI_BASE_URL")
            and os.getenv("TEXT_MODEL")
            and os.getenv("VISION_MODEL")
            and os.getenv("EMBEDDING_MODEL")
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
