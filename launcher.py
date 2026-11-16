"""VoiceTwin 启动器（PyInstaller 入口）。"""

from __future__ import annotations

import sys


def main() -> int:
    from voicetwin.cli import main as cli_main

    argv = sys.argv[1:]
    if not argv:
        argv = ["ui", "--port", "8766"]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
