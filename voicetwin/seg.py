"""低延迟分段：用端点检测替代固定窗口。

设计要点：
- 采集线程只负责 ``push(pcm)``，判定在另一个线程按 20ms 帧推进；
- 静音 **不出块**，所以 CPU 在没人说话时接近 0；
- 说完后静音 ``min_silence`` 秒立刻封口，延迟不再由窗口长度决定；
- 单句超过 ``max_segment`` 秒强制切断，避免长段拖死延迟。

单位约定：所有时间参数是秒，采样率默认 16000，PCM 为 16bit 单声道小端。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

RATE = 16000
FRAME_MS = 20
FRAME_BYTES = RATE * 2 * FRAME_MS // 1000


def frame_rms(frame: bytes) -> float:
    """一帧 16bit PCM 的 RMS（0..1）。"""
    if not frame:
        return 0.0
    count = len(frame) // 2
    if count <= 0:
        return 0.0
    total = 0
    for (value,) in struct.iter_unpack("<h", frame[: count * 2]):
        total += value * value
    return math.sqrt(total / count) / 32768.0


@dataclass
class SegmentSpan:
    """一段已完成的话。"""

    pcm: bytes
    start: float
    end: float
    voiced: float = 0.0

    @property
    def seconds(self) -> float:
        return self.end - self.start


class RingBuffer:
    """保留最近 N 秒音频，供草稿与补头使用。"""

    def __init__(self, seconds: float = 30.0, rate: int = RATE) -> None:
        self.max_bytes = int(seconds * rate * 2)
        self._buf = bytearray()
        self._pushed = 0

    def push(self, pcm: bytes) -> None:
        if not pcm:
            return
        self._buf.extend(pcm)
        self._pushed += len(pcm)
        if len(self._buf) > self.max_bytes:
            del self._buf[: len(self._buf) - self.max_bytes]

    @property
    def pushed(self) -> int:
        """累计推入的字节数（单调递增，可用作时间轴刻度）。"""
        return self._pushed

    def tail(self, seconds: float) -> bytes:
        size = max(0, int(seconds * RATE * 2))
        return bytes(self._buf[-size:]) if size else b""

    def since(self, mark: int) -> bytes:
        """取 mark 之后推送的数据；mark 太旧时返回当前可用的全部。"""
        offset = self._pushed - len(self._buf)
        start = max(mark - offset, 0)
        return bytes(self._buf[start:])


class Segmenter:
    """帧级端点检测器。

    ``feed`` 可以以任意块大小调用，内部按 20ms 对齐；返回所有已封口的段。
    """

    def __init__(
        self,
        min_speech: float = 0.15,
        max_segment: float = 2.5,
        hard_max: float = 4.5,
        min_silence: float = 0.20,
        energy_floor: float = 0.012,
        preroll: float = 0.12,
        rate: int = RATE,
    ) -> None:
        self.rate = rate
        self.frame_bytes = rate * 2 * FRAME_MS // 1000
        self.min_speech = min_speech
        self.max_segment = max_segment
        self.hard_max = max(hard_max, max_segment)
        self.min_silence = min_silence
        self.energy_floor = energy_floor
        self.preroll_frames = max(0, int(round(preroll * 1000 / FRAME_MS)))
        self._pending = bytearray()
        self._preroll: list[bytes] = []
        self._run: list[bytes] = []
        self._levels: list[float] = []
        self._run_start = 0.0
        self._elapsed = 0.0
        self._voiced = 0.0
        self._quiet = 0.0
        self._active = False
        self._offset = 0.0  # 已消费音频的秒数（块起点）

    # ---------- 对外 ----------

    def feed(self, pcm: bytes) -> list[SegmentSpan]:
        if pcm:
            self._pending.extend(pcm)
        out: list[SegmentSpan] = []
        fb = self.frame_bytes
        while len(self._pending) >= fb:
            frame = bytes(self._pending[:fb])
            del self._pending[:fb]
            span = self._step(frame)
            if span is not None:
                out.append(span)
        return out

    def flush(self) -> SegmentSpan | None:
        """把尾巴强制封口（停止监听时调用）。"""
        if self._pending:
            frame = bytes(self._pending)
            self._pending.clear()
            self._step(frame)
        if not self._run:
            self._reset()
            return None
        span = self._close()
        self._reset()
        return span

    @property
    def in_speech(self) -> bool:
        return self._active

    def partial_window(self, seconds: float = 1.2) -> bytes:
        """草稿用：当前句已累积的音频（不足则全给）。"""
        data = b"".join(self._run)
        size = int(seconds * self.rate * 2)
        return data if len(data) <= size else data[-size:]

    # ---------- 内部 ----------

    def _step(self, frame: bytes) -> SegmentSpan | None:
        level = frame_rms(frame)
        self._elapsed += FRAME_MS / 1000
        voiced = level >= self.energy_floor

        if not self._active:
            if voiced:
                self._active = True
                self._voiced = FRAME_MS / 1000
                self._quiet = 0.0
                self._run = list(self._preroll) + [frame]
                self._levels = [0.0] * len(self._preroll) + [level]
                self._run_start = max(0.0, self._elapsed - FRAME_MS / 1000 - len(self._preroll) * FRAME_MS / 1000)
                self._preroll.clear()
            else:
                self._preroll.append(frame)
                if len(self._preroll) > self.preroll_frames:
                    self._preroll.pop(0)
            return None

        self._run.append(frame)
        self._levels.append(level)
        if voiced:
            self._voiced += FRAME_MS / 1000
            self._quiet = 0.0
        else:
            self._quiet += FRAME_MS / 1000

        duration = len(self._run) * FRAME_MS / 1000
        if self._quiet >= self.min_silence and self._voiced >= self.min_speech:
            span = self._close(trim_frames=int(self._quiet * 1000 / FRAME_MS))
            self._reset()
            return span
        if duration >= self.max_segment:
            # 到点了先找一处静音再切；找不到就再等一会儿（最多 hard_max），
            # 硬切会把词从中间劈开，宁可多等半秒。
            cut = self._best_cut()
            if cut:
                span = self._close_at(cut)
                self._run = self._run[cut:]
                self._levels = self._levels[cut:]
                self._run_start = span.end
                self._quiet = 0.0
                return span
            if duration >= self.hard_max:
                span = self._close()
                self._reset()
                return span
            return None
        return None

    def _best_cut(self) -> int:
        """在最近 0.8s 里找一段「像样的静音」（≥120ms）切；没有就返回 0。"""
        tail = int(0.8 * 1000 / FRAME_MS)
        floor = max(1, len(self._run) - tail)
        need = max(3, int(0.12 * 1000 / FRAME_MS))
        best = 0
        i = len(self._run) - 1
        while i >= floor:
            if self._levels[i] < self.energy_floor:
                start = i
                while start > floor and self._levels[start - 1] < self.energy_floor:
                    start -= 1
                if (i - start + 1) >= need and start * FRAME_MS / 1000 >= self.min_speech:
                    return start
                i = start - 1
            else:
                i -= 1
        return best

    def _close_at(self, index: int) -> SegmentSpan:
        frames = self._run[:index]
        pcm = b"".join(frames)
        end = self._run_start + len(frames) * FRAME_MS / 1000
        return SegmentSpan(pcm=pcm, start=self._run_start, end=end, voiced=self._voiced)

    def _close(self, trim_frames: int = 0) -> SegmentSpan:
        frames = self._run[: len(self._run) - trim_frames] if trim_frames else self._run
        pcm = b"".join(frames)
        end = self._run_start + len(frames) * FRAME_MS / 1000
        return SegmentSpan(pcm=pcm, start=self._run_start, end=end, voiced=self._voiced)

    def _reset(self) -> None:
        self._active = False
        self._run = []
        self._levels = []
        self._voiced = 0.0
        self._quiet = 0.0


def overlap_trim(previous: str, current: str) -> str:
    """裁掉与上一句重复的词前缀（重叠窗口的残留）。"""
    a = previous.strip()
    b = current.strip()
    if not a or not b:
        return b
    aw = a.split()
    bw = b.split()
    limit = min(len(aw), len(bw), 8)
    for size in range(limit, 0, -1):
        if [w.lower() for w in aw[-size:]] == [w.lower() for w in bw[:size]]:
            return " ".join(bw[size:])
    # 中文没有空格：按字符尾巴比较
    if " " not in a and " " not in b:
        for size in range(min(len(a), len(b), 12), 0, -1):
            if a[-size:] == b[:size]:
                return b[size:]
    return b
