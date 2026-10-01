"""让验收脚本在 Windows 控制台安全输出 PDF 和模型文本。"""

from __future__ import annotations

import sys


def configure_utf8_stdout() -> None:
    """优先使用 UTF-8，避免特殊连字等字符触发 GBK 编码异常。"""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")
