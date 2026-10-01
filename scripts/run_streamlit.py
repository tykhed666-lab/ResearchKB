"""供 PyCharm 运行按钮使用的 Streamlit 启动入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli

# app.py 位于项目根目录；使用绝对路径可避免工作目录变化导致找不到文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


def main() -> None:
    """按照 ``streamlit run app.py`` 的方式启动网页。"""
    sys.argv = ["streamlit", "run", str(APP_PATH)]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
