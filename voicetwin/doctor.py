"""本机环境体检：告诉你缺什么、怎么补。"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request

from .config import Config


def check_python() -> tuple[str, str, bool, str]:
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
    hint = "OK" if ok else "建议用 3.11/3.12 建独立 venv（ML wheel 覆盖更全）"
    return "python", f"{v.major}.{v.minor}.{v.micro}", ok, hint


def check_ffmpeg() -> tuple[str, str, bool, str]:
    path = shutil.which("ffmpeg")
    return "ffmpeg", path or "未安装", bool(path), "winget install Gyan.FFmpeg" if not path else "OK"


def check_ollama(url: str) -> tuple[str, str, bool, str]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=3) as resp:
            models = [m.get("name", "?") for m in json.loads(resp.read()).get("models", [])]
        return "ollama", f"{len(models)} 个模型", True, ", ".join(models[:4]) or "OK"
    except Exception as exc:  # noqa: BLE001
        return "ollama", f"不可达（{type(exc).__name__}）", False, f"启动 ollama serve，确认 {url}"


def check_gpu() -> tuple[str, str, bool, str]:
    if not shutil.which("nvidia-smi"):
        return "gpu", "未检出 NVIDIA 显卡", False, "实时/克隆模式需要 GPU；否则先用离线文件版"
    return "gpu", "已检出 nvidia-smi", True, "OK"


def run_all(cfg: Config | None = None) -> list[tuple[str, str, bool, str]]:
    cfg = cfg or Config()
    return [check_python(), check_ffmpeg(), check_ollama(cfg.ollama_url), check_gpu()]
