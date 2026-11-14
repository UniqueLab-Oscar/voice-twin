"""在线说话人区分：不用下载模型，靠基频（音高）聚类，够区分男女与不同嗓音。

实时场景拿不到整段音频，pyannote 那种全片段聚类用不上；这里做的是
「来一块、判一次」的在线判断：

1. 每块语音算中位基频 F0（自相关）与能量；
2. 太轻的块直接沿用上一位说话人（避免噪声乱开新说话人）；
3. 与已有说话人中心的音高差 < 12% 视为同一人，否则开新说话人（上限 8 个）。

局限写在文档里：音高接近的两个同性别说话人会合并，正式版应换成
ECAPA 声纹向量（见 docs/ARCHITECTURE.md §15）。
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field


def pcm_to_floats(pcm: bytes) -> list[float]:
    """16bit 单声道小端 PCM → [-1, 1] 浮点。"""

    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    return [s / 32768.0 for s in samples]


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def median_f0(samples: list[float], rate: int, lo: float = 70.0, hi: float = 400.0) -> float:
    """自相关估基频；返回 0 表示没找到周期（静音/噪声）。"""

    n = len(samples)
    if n < rate // 20:  # 少于 50ms 无法估
        return 0.0
    min_lag = int(rate / hi)
    max_lag = min(int(rate / lo), n - 1)
    if max_lag <= min_lag:
        return 0.0

    best_lag, best_score = 0, 0.0
    energy = sum(s * s for s in samples) or 1e-9
    for lag in range(min_lag, max_lag):
        acc = 0.0
        for i in range(n - lag):
            acc += samples[i] * samples[i + lag]
        score = acc / energy
        if score > best_score:
            best_score, best_lag = score, lag
    if best_score < 0.25 or best_lag == 0:
        return 0.0
    return rate / best_lag


@dataclass
class Speaker:
    name: str
    f0: float
    hits: int = 0
    ref_pcm: bytes = b""
    ref_rate: int = 16000


@dataclass
class SpeakerTracker:
    """在线说话人分配器。"""

    tolerance: float = 0.20      # 音高差 20% 以内算同一人（同一人跨句音高会漂）
    min_rms: float = 0.01        # 低于这个能量不判新说话人
    max_speakers: int = 8
    speakers: list[Speaker] = field(default_factory=list)
    confirm_hits: int = 2        # 连续几块都像新声音才真的开新说话人
    _last: int = 0
    _pending: tuple[float, int] | None = None

    def assign(self, pcm: bytes, rate: int = 16000) -> str:
        samples = pcm_to_floats(pcm)
        level = rms(samples)
        f0 = median_f0(samples, rate)

        if level < self.min_rms or f0 <= 0:
            # 静音或噪声：沿用上一位，字幕不至于乱跳
            return self.speakers[self._last].name if self.speakers else "S0"

        for i, sp in enumerate(self.speakers):
            if abs(f0 - sp.f0) / max(sp.f0, 1.0) <= self.tolerance:
                sp.hits += 1
                sp.f0 = (sp.f0 * (sp.hits - 1) + f0) / sp.hits
                self._last = i
                if not sp.ref_pcm:
                    sp.ref_pcm, sp.ref_rate = pcm, rate
                return sp.name

        # 新声音要连续出现几块才确认，避免一次变调就多出一个人
        if self._pending and abs(f0 - self._pending[0]) / max(self._pending[0], 1.0) <= self.tolerance:
            self._pending = (self._pending[0], self._pending[1] + 1)
        else:
            self._pending = (f0, 1)

        if len(self.speakers) < self.max_speakers and self._pending[1] >= self.confirm_hits:
            self._pending = None
            name = f"S{len(self.speakers)}"
            self.speakers.append(Speaker(name=name, f0=f0, hits=1, ref_pcm=pcm, ref_rate=rate))
            self._last = len(self.speakers) - 1
            return name

        if not self.speakers:
            # 还没有任何说话人档案：先按 S0 出字幕，等确认后再建档
            return "S0"

        # 说话人满了：归到音高最近的一位
        nearest = min(range(len(self.speakers)), key=lambda i: abs(f0 - self.speakers[i].f0))
        self._last = nearest
        return self.speakers[nearest].name

    def reference(self, name: str) -> tuple[bytes, int]:
        for sp in self.speakers:
            if sp.name == name:
                return sp.ref_pcm, sp.ref_rate
        return b"", 16000

    def summary(self) -> str:
        if not self.speakers:
            return "（尚未识别到说话人）"
        return "，".join(f"{s.name} ≈ {s.f0:.0f} Hz" for s in self.speakers)
