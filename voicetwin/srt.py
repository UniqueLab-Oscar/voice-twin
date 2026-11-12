"""字幕导出：把译员结果写成 SRT，便于直接挂到播放器。"""

from __future__ import annotations

from .types import RunReport


def _ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(report: RunReport) -> str:
    blocks: list[str] = []
    for i, syn in enumerate(report.syntheses, start=1):
        seg = syn.translated.segment
        blocks.append(f"{i}\n{_ts(seg.start)} --> {_ts(seg.end)}\n{syn.translated.text}\n")
    return "\n".join(blocks)
