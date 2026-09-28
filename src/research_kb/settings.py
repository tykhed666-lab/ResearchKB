"""集中保存多个业务模块共用的默认参数。"""

# 一次问答最多取回多少条证据；它不是相似度阈值。
DEFAULT_TOP_K = 5

# 文本切分参数的单位是字符，不是模型 token。
DEFAULT_CHUNK_SIZE = 800

# 相邻片段重复的字符数，帮助保留跨边界的上下文。
DEFAULT_CHUNK_OVERLAP = 120