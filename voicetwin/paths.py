"""路径解析：源码运行和 PyInstaller 打包后都能找到模型与示例。

PyInstaller 6 的 onedir 布局把数据放进 ``_internal``（运行时是 ``sys._MEIPASS``），
exe 本身在上一级，所以两个位置都要看。
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """模型/示例所在的目录。"""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            base = Path(meipass)
            if (base / "models").exists():
                return base
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (exe_dir, exe_dir / "_internal"):
            if (candidate / "models").exists():
                return candidate
        return exe_dir
    return Path(__file__).resolve().parent.parent


def model_dir(name: str = "st-zh-en") -> Path:
    return app_root() / "models" / name


def demo_dir() -> Path:
    return app_root() / "demo"
