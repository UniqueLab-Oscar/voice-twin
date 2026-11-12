"""报告渲染：Markdown 表格 + 机器可读的 run.json。

单独成模块，避免 CLI 与 Web 服务互相 import 成环。
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import RunReport


def _fmt_ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def render_report(report: RunReport) -> str:
    lines = [
        "# VoiceTwin 运行报告",
        "",
        "> 译文由「译员式」重述生成；音色由克隆 TTS 保留。",
        "",
        "| # | 时间 | 说话人 | 原文（ASR） | 译员版 | 音频 |",
        "|---|---|---|---|---|---|",
    ]
    for i, syn in enumerate(report.syntheses, start=1):
        seg = syn.translated.segment
        audio = Path(syn.audio_path).name if syn.audio_path else "—"
        lines.append(
            f"| {i} | {_fmt_ts(seg.start)} | {seg.speaker} | {seg.text} | {syn.translated.text} | {audio} |"
        )
    lines += ["", "## 耗时", ""]
    for k, v in report.elapsed_ms.items():
        lines.append(f"- {k}: {v:.1f} ms" if v >= 0 else f"- {k}: 失败")
    return "\n".join(lines) + "\n"


def dump_run(report: RunReport, path: Path) -> None:
    """机器可读的中间产物，便于离线重放与调参。"""
    data = {
        "segments": [s.__dict__ for s in report.segments],
        "translations": [
            {
                "start": t.segment.start,
                "end": t.segment.end,
                "speaker": t.segment.speaker,
                "text": t.text,
            }
            for t in report.translations
        ],
        "elapsed_ms": report.elapsed_ms,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
