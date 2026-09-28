"""集中保存多个业务模块共用的默认参数。"""

from pathlib import Path


# settings.py 位于：
# 项目根目录/src/research_kb/settings.py
# parents[2] 因此对应 ResearchKB 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 项目运行时产生的数据统一保存在 data 目录。
DATA_DIR = PROJECT_ROOT / "data"

# SQLite 文档注册表的位置。
DOCUMENT_DATABASE_PATH = (
    DATA_DIR / "research_kb.db"
)

# 一次问答最多取回多少条证据；它不是相似度阈值。
DEFAULT_TOP_K = 5

# 文本切分参数的单位是字符，不是模型 token。
DEFAULT_CHUNK_SIZE = 800

# 相邻片段重复的字符数，帮助保留跨边界的上下文。
DEFAULT_CHUNK_OVERLAP = 120
