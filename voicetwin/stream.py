"""分块同传：把长音频切成短块，边切边译，量化「延迟」而不只是总耗时。

离线整段处理的指标是「跑完要多久」；同传真正关心的指标是
「这块音频说完之后，多久能看到中文」。本模块用固定窗口切块，
按块的顺序逐块出结果，并记录每一块的：

- `latency_ms`：该块音频从开始识别到译文可用的耗时；
- `lag_ms`：处理完该块时，处理进度落后音频多少毫秒（正数=落后，负数=领先）；
- `rtf`：总处理耗时 / 音频总时长，小于 1 才具备实时性。

切块在时间轴上做，不依赖 ffmpeg，纯标准库 wave。
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .types import Segment, Translated

MIN_CHUNK_SECONDS = 0.4


@dataclass
class Chunk:
    """一段被切出来的音频。"""

    index: int
    start: float
    end: float
    path: Path

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class StreamEvent:
    """一块音频的处理结果。"""

    chunk: Chunk
    segments: list[Segment]
    translations: list[Translated]
    latency_ms: float
    lag_ms: float


@dataclass
class StreamReport:
    events: list[StreamEvent] = field(default_factory=list)
    audio_seconds: float = 0.0
    wall_ms: float = 0.0

    @property
    def rtf(self) -> float:
        if self.audio_seconds <= 0:
            return 0.0
        return (self.wall_ms / 1000.0) / self.audio_seconds

    @property
    def first_result_ms(self) -> float:
        return self.events[0].latency_ms if self.events else 0.0

    @property
    def max_lag_ms(self) -> float:
        return max((e.lag_ms for e in self.events), default=0.0)

    @property
    def translations(self) -> list[Translated]:
        out: list[Translated] = []
        for ev in self.events:
            out.extend(ev.translations)
        return out


def split_wav(path: str | Path, out_dir: str | Path, window: float = 3.0) -> list[Chunk]:
    """按固定时长切 wav；最后不足 `MIN_CHUNK_SECONDS` 的尾巴并进上一块。"""

    src = Path(path)
    if window <= 0:
        raise ValueError("window 必须大于 0")
    dst_dir = Path(out_dir) / "chunks"
    dst_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    with wave.open(str(src), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        total = w.getnframes()
        step = max(1, int(round(window * rate)))

        # 先定好边界：最后一块太短就并进上一块，避免切出没有语音的碎片
        bounds: list[tuple[int, int]] = []
        pos = 0
        while pos < total:
            end = min(pos + step, total)
            bounds.append((pos, end))
            pos = end
        if len(bounds) > 1 and (bounds[-1][1] - bounds[-1][0]) / rate < MIN_CHUNK_SECONDS:
            _, tail_end = bounds.pop()
            head_start, _ = bounds[-1]
            bounds[-1] = (head_start, tail_end)

        for index, (start, end) in enumerate(bounds, start=1):
            w.setpos(start)
            data = w.readframes(end - start)
            out = dst_dir / f"{index:03d}.wav"
            with wave.open(str(out), "wb") as o:
                o.setnchannels(channels)
                o.setsampwidth(width)
                o.setframerate(rate)
                o.writeframes(data)
            chunks.append(Chunk(index, start / rate, end / rate, out))
    return chunks


class StreamRunner:
    """按顺序处理音频块；识别与翻译用注入的函数，方便测试。"""

    def __init__(
        self,
        asr_fn: Callable[[str, str], list[Segment]],
        translate_fn: Callable[[list[Segment]], list[Translated]],
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> None:
        self.asr_fn = asr_fn
        self.translate_fn = translate_fn
        self.on_event = on_event

    def run(self, chunks: Sequence[Chunk]) -> StreamReport:
        report = StreamReport(audio_seconds=sum(c.seconds for c in chunks))
        t_all = time.perf_counter()
        consumed = 0.0
        prompt = ""

        for chunk in chunks:
            consumed += chunk.seconds
            t0 = time.perf_counter()
            segments = self.asr_fn(str(chunk.path), prompt)
            for seg in segments:
                # 块内时间是相对坐标，挪回整段音频的绝对时间轴
                seg.start += chunk.start
                seg.end += chunk.start
            if segments:
                # 把已识别文本的尾巴喂给下一块，跨块句子才接得上
                prompt = (prompt + " " + segments[-1].text).strip()[-240:]
            translations = self.translate_fn(segments) if segments else []
            latency = (time.perf_counter() - t0) * 1000
            wall = (time.perf_counter() - t_all) * 1000
            # 正数=处理落后于音频（积压），负数=处理跑在音频前面
            event = StreamEvent(chunk, segments, translations, latency, wall - consumed * 1000)
            report.events.append(event)
            if self.on_event:
                self.on_event(event)

        report.wall_ms = (time.perf_counter() - t_all) * 1000
        return report


def _ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _lag_text(lag_ms: float) -> str:
    if lag_ms >= 0:
        return f"落后 {lag_ms:.0f} ms"
    return f"领先 {-lag_ms:.0f} ms"


def render_srt(report: StreamReport) -> str:
    blocks: list[str] = []
    for i, tr in enumerate(report.translations, start=1):
        seg = tr.segment
        blocks.append(f"{i}\n{_ts(seg.start)} --> {_ts(seg.end)}\n{tr.text}\n")
    return "\n".join(blocks)


def render_stream(report: StreamReport, window: float) -> str:
    lines = [
        "# 分块同传延迟报告",
        "",
        f"- 音频总长：{report.audio_seconds:.2f}s",
        f"- 分块窗口：{window:.2f}s",
        f"- 处理总耗时：{report.wall_ms / 1000:.2f}s",
        f"- RTF（耗时/音频）：{report.rtf:.2f}"
        + ("（可实时）" if report.rtf < 1 else "（跟不上，需提速）"),
        f"- 首块延迟：{report.first_result_ms:.0f} ms",
        f"- 最大落后：{report.max_lag_ms:.0f} ms" if report.max_lag_ms > 0 else "- 全程领先：处理快于音频输入",
        "",
        "| 块 | 音频区间 | 块时长 | 延迟 | 进度 | 译文 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for ev in report.events:
        text = " / ".join(t.text for t in ev.translations) or "（无语音）"
        lines.append(
            f"| {ev.chunk.index} | {ev.chunk.start:.1f}-{ev.chunk.end:.1f}s | "
            f"{ev.chunk.seconds:.1f}s | {ev.latency_ms:.0f} ms | {_lag_text(ev.lag_ms)} | {text} |"
        )
    return "\n".join(lines) + "\n"
