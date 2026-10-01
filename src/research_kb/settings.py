"""集中管理项目路径、环境变量和业务默认值。

业务模块不应各自推导项目根目录或重复读取 ``.env``。统一入口可以让
配置来源更容易理解，也便于测试时通过环境变量覆盖本地配置。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# settings.py 位于：
# 项目根目录/src/research_kb/settings.py
# parents[2] 因此对应 ResearchKB 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# 项目运行时产生的数据统一保存在 data 目录。
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGE_ROOT = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
EXTERNAL_HTML_DIR = DATA_DIR / "external" / "sec"

# SQLite 文档注册表的位置。
DOCUMENT_DATABASE_PATH = DATA_DIR / "research_kb.db"

# 一次问答最多取回多少条证据；它不是相似度阈值。
DEFAULT_TOP_K = 5

# 文本切分参数的单位是字符，不是模型 token。
DEFAULT_CHUNK_SIZE = 800

# 相邻片段重复的字符数，帮助保留跨边界的上下文。
DEFAULT_CHUNK_OVERLAP = 120


def load_environment() -> None:
    """加载项目根目录的 ``.env``，但不覆盖进程已有环境变量。

    ``python-dotenv`` 默认不会覆盖已经设置的值。这样既支持本地开发的
    ``.env``，也允许 CI、容器或测试通过系统环境变量注入配置。
    """
    load_dotenv(ENV_PATH, override=False)


def require_environment(*names: str) -> tuple[str, ...]:
    """读取一组必填环境变量，并按传入顺序返回清理后的值。

    Args:
        names: 需要读取的环境变量名。

    Raises:
        ValueError: 没有传入变量名，或任一变量未配置。
    """
    if not names:
        raise ValueError("至少需要指定一个环境变量名")

    load_environment()
    values = tuple((os.getenv(name) or "").strip() for name in names)
    missing = [name for name, value in zip(names, values, strict=True) if not value]

    if missing:
        raise ValueError(f"缺少环境变量：{', '.join(missing)}")

    return values


def optional_environment(name: str, default: str = "") -> str:
    """读取可选环境变量；空值会回退到 ``default``。"""
    if not name.strip():
        raise ValueError("环境变量名不能为空")

    load_environment()
    return (os.getenv(name) or default).strip()
