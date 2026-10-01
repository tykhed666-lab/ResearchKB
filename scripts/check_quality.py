"""按固定顺序执行项目的本地质量门禁。

运行方式：``uv run python scripts/check_quality.py``。
任何一步失败都会立即返回非零退出码，GitHub Actions 与本地开发使用同一套
检查，避免出现“本机能过、CI 失败”的规则差异。
"""

from __future__ import annotations

import subprocess
import sys

PYTHON_TARGETS = ("app.py", "src", "tests", "scripts")


def run_step(name: str, command: list[str]) -> None:
    """执行一个质量检查步骤，并把命令原样展示给开发者。"""
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    """依次检查格式、静态规则、语法编译和自动化测试。"""
    run_step(
        "格式检查",
        ["ruff", "format", "--check", *PYTHON_TARGETS],
    )
    run_step(
        "静态检查",
        ["ruff", "check", *PYTHON_TARGETS],
    )
    run_step(
        "语法编译",
        [sys.executable, "-m", "compileall", "-q", *PYTHON_TARGETS],
    )
    run_step("自动化测试", ["pytest", "-q"])
    print("\n全部质量检查通过。")


if __name__ == "__main__":
    main()
