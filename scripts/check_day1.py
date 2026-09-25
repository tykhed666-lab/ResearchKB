from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pypdf import PdfReader
from pymilvus import MilvusClient


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def check_packages() -> dict[str, str]:
    """读取关键依赖的实际安装版本。"""
    names = [
        "langchain",
        "langgraph",
        "langchain-milvus",
        "langchain-openai",
        "langchain-ollama",
        "langchain-mcp-adapters",
        "mcp",
        "pymilvus",
        "pymupdf",
        "pypdf",
        "streamlit",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def check_pdfs() -> dict[str, object]:
    """统计原始 PDF 数量和每份文档的页数。"""
    # RAW_DIR.glob("*.pdf"):按通配符匹配目录下所有以 .pdf 结尾的文件,返回一个生成器,每个元素是 Path 对象。对应你项目里的 5 份文档(NVDA、AMD、LITE、MU、STX)
    #sorted(...):按文件名排序,保证每次运行输出顺序一致(便于对比和复查)
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


def check_ollama(base_url: str, model: str) -> dict[str, object]:
    """检查本地模型列表，并执行一次最小文本生成。"""
    with httpx.Client(base_url=base_url, timeout=90) as client:
        tags_response = client.get("/api/tags")
        tags_response.raise_for_status()
        installed = [item["name"] for item in tags_response.json().get("models", [])]
        generate_response = client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": "只回复 DAY1_OK",
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
            },
        )
        generate_response.raise_for_status()
    return {
        "base_url": base_url,
        "installed_models": installed,
        "smoke_response": generate_response.json().get("response", "").strip(),
    }


def main() -> int:
    """汇总第一天的环境、数据、数据库和本地模型检查结果。"""
    load_dotenv(ROOT / ".env")
    milvus_uri = os.getenv("MILVUS_URI", "http://192.168.88.161:19530")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model = os.getenv("OLLAMA_TEXT_MODEL", "qwen3:1.7b")

    # 只输出配置是否齐全，不输出任何 API Key。
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "packages": check_packages(),
        "pdfs": check_pdfs(),
        "milvus": check_milvus(milvus_uri),
        "ollama": check_ollama(ollama_base_url, ollama_model),
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
