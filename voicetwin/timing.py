"""把合成语音对齐回原说话人的时间轴（VoiceTwin 自有算法）。

同传的难点不是「译得准」，而是**译完还能跟得上**：同一句话，中文比英文短，
机器合成的配音如果按自己的节奏播完，就会和画面/嘴型越差越远。

这里的做法完全基于标准库，不依赖 ffmpeg：

1. 用 10 ms 帧的能量（RMS）找出首尾静音，先剪掉「没人说话」的部分；
2. 剪完仍比目标长 → 线性重采样加速（有上限，避免花栗鼠音），超出的尾部淡出截断；
3. 比目标短 → 尾部补静音，让时间轴严格对齐（下游混音/对齐不需要再算漂移）。

加速会整体升调，这是刻意的取舍：口语同传里「跟上」比「音高绝对不变」更重要，
上限 ``max_speed`` 保证听感不变形。
"""

from __future__ import annotations

import wave
from array import array
from pathlib import Path

FRAME_SECONDS = 0.01  # 10 ms 一帧，够细又够快


def _read_mono(path: Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"只支持 16-bit PCM wav，当前 {width * 8}-bit：{path.name}")
    samples = array("h")
    samples.frombytes(raw)
    if channels > 1:  # 取左声道即可，配音本身就是单声道
        samples = samples[::channels]
    return samples, rate


def _write_mono(path: Path, samples: array, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def _frame_rms(samples: array, frame: int) -> list[float]:
    out: list[float] = []
    for start in range(0, len(samples), frame):
        chunk = samples[start : start + frame]
        if not chunk:
            break
        out.append((sum(v * v for v in chunk) / len(chunk)) ** 0.5)
    return out


def trim_silence(samples: array, rate: int, threshold_ratio: float = 0.02) -> array:
    """剪掉首尾静音，阈值相对于本段最大能量，避免不同录音音量差异。"""
    frame = max(1, int(rate * FRAME_SECONDS))
    rms = _frame_rms(samples, frame)
    if not rms:
        return samples
    peak = max(rms) or 1.0
    threshold = peak * threshold_ratio
    first = next((i for i, v in enumerate(rms) if v >= threshold), 0)
    last = next((i for i in range(len(rms) - 1, -1, -1) if rms[i] >= threshold), len(rms) - 1)
    return samples[first * frame : (last + 1) * frame]


def _resample(samples: array, factor: float) -> array:
    """线性插值重采样。factor > 1 表示变短（加速）。"""
    if abs(factor - 1.0) < 1e-9 or not samples:
        return samples
    out_len = max(1, int(len(samples) / factor))
    src = len(samples) - 1
    out = array("h", bytes(2 * out_len))
    for i in range(out_len):
        pos = i * factor
        i0 = int(pos)
        if i0 >= src:
            out[i] = samples[src]
            continue
        frac = pos - i0
        out[i] = int(samples[i0] * (1 - frac) + samples[i0 + 1] * frac)
    return out


def _fade_out(samples: array, rate: int, ms: int = 10) -> array:
    n = min(len(samples), int(rate * ms / 1000))
    for i in range(n):
        samples[len(samples) - n + i] = int(samples[len(samples) - n + i] * (1 - i / n))
    return samples


def fit_to_timeline(
    src: str | Path,
    dst: str | Path,
    target_seconds: float,
    max_speed: float = 1.3,
    pad_with_silence: bool = True,
) -> Path:
    """把 `src` 对齐成「恰好占满 target_seconds」的 `dst`，返回 dst。"""
    src, dst = Path(src), Path(dst)
    target_seconds = max(0.05, float(target_seconds))
    samples, rate = _read_mono(src)
    samples = trim_silence(samples, rate)
    target_len = max(1, int(target_seconds * rate))

    if len(samples) > target_len:
        speed = len(samples) / target_len
        speed = min(speed, max_speed)
        samples = _resample(samples, speed)
        if len(samples) > target_len:
            samples = samples[:target_len]
            _fade_out(samples, rate)
    elif len(samples) < target_len and pad_with_silence:
        samples.extend(array("h", bytes(2 * (target_len - len(samples)))))

    _write_mono(dst, samples, rate)
    return dst


def duration_seconds(path: str | Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def is_wav(path: str | Path) -> bool:
    """廉价探测：外部引擎可能返回 mp3/flac 或错误页，后处理前先确认能解析。"""
    try:
        with wave.open(str(path), "rb"):
            return True
    except (wave.Error, EOFError, OSError):
        return False
