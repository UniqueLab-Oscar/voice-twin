"""播放层：常开输出流 + 短队列，做到「译完就出声」。

设计：
- ``OutputStream`` 常开，避免每句都重新打开设备（Windows 上是几百毫秒的开销）；
- 队列里按块写，首块 80ms 就够，不必等整句合成完；
- 新句来时可选「打断」：旧句 200ms 淡出后清空队列。
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np

RATE = 24000
CHUNK_MS = 80


def _fade_out(samples: np.ndarray, ms: int = 200, rate: int = RATE) -> np.ndarray:
    n = min(len(samples), int(rate * ms / 1000))
    if n <= 0:
        return samples
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out = samples.astype(np.float32).copy()
    out[-n:] *= ramp
    return out


def normalize(samples: np.ndarray, peak: float = 0.7) -> np.ndarray:
    """峰值归一化到 peak，避免忽大忽小。"""
    if samples.size == 0:
        return samples
    current = float(np.max(np.abs(samples)))
    if current <= 1e-6:
        return samples
    return (samples * (peak / current)).astype(np.float32)


class Player:
    """把 PCM(int16) 丢进队列，后台线程写给声卡。"""

    def __init__(self, rate: int = RATE, volume: float = 1.0, device: int | None = None) -> None:
        self.rate = rate
        self.volume = max(0.0, float(volume))
        self.device = device
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream = None
        self.played_ms = 0
        self.dropped = 0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="vt-player")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.close()
            finally:
                self._stream = None

    # ---------- 接口 ----------

    def enqueue(self, pcm: bytes, interrupt: bool = False) -> None:
        if interrupt:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        for i in range(0, len(pcm), self.rate * 2 * CHUNK_MS // 1000):
            self._queue.put(pcm[i : i + self.rate * 2 * CHUNK_MS // 1000])

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    # ---------- 内部 ----------

    def _run(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=int(self.rate * CHUNK_MS / 1000),
        )
        self._stream.start()
        try:
            while not self._stop.is_set():
                try:
                    block = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if block is None:
                    break
                data = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
                if self.volume != 1.0:
                    data = data * self.volume
                self._stream.write(np.clip(data, -1.0, 1.0).astype(np.float32))
                self.played_ms += int(len(data) / self.rate * 1000)
        finally:
            try:
                self._stream.stop()
            except Exception:  # noqa: BLE001
                pass


def has_output_device() -> bool:
    try:
        import sounddevice as sd

        return any(dev["max_output_channels"] > 0 for dev in sd.query_devices())
    except Exception:  # noqa: BLE001
        return False


def list_output_devices() -> str:
    import sounddevice as sd

    lines = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            lines.append(f"[{i}] {dev['name']}")
    return "\n".join(lines) or "（没有输出设备）"
