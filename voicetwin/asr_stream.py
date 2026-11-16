"""流式识别（sherpa-onnx）：边说边出字，端点自己判，不用等一句说完。

和 faster-whisper 的差别：
- whisper 是「一段音频进、整句出」，必须等静音封口；
- sherpa 的 OnlineRecognizer 是「持续喂样本、随时读当前假设」，说完立刻定稿。

代价是模型固定（这里用中英双语 zipformer），换来 200–400ms 的跟随感。
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

RATE = 16000


def _preload_onnxruntime() -> None:
    """Windows 上 System32 里常有一个旧版 onnxruntime.dll（1.17），会先被加载。

    sherpa-onnx 需要更新的 ORT API（28），所以先把 venv 里那份按全路径加载进来，
    Windows 按模块名缓存，后续 sherpa 的 import 就会复用它。
    """

    if sys.platform != "win32":
        return
    try:
        import ctypes
        import os

        import onnxruntime

        capi = Path(onnxruntime.__file__).parent / "capi"
        if not (capi / "onnxruntime.dll").exists():
            return
        os.add_dll_directory(str(capi))
        ctypes.WinDLL(str(capi / "onnxruntime.dll"))
    except Exception:  # noqa: BLE001
        pass


class StreamAsr:
    """薄封装 sherpa-onnx 的在线识别器。"""

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 4,
        provider: str = "cpu",
        decoding: str = "greedy_search",
        rule1: float = 0.6,
        rule2: float = 0.35,
    ) -> None:
        _preload_onnxruntime()
        import sherpa_onnx

        base = Path(model_dir)
        tokens = base / "tokens.txt"
        encoder = _pick(base, "encoder-epoch-99-avg-1.int8.onnx", "encoder-epoch-99-avg-1.onnx")
        decoder = _pick(base, "decoder-epoch-99-avg-1.onnx")
        joiner = _pick(base, "joiner-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.onnx")
        for path in (tokens, encoder, decoder, joiner):
            if not path.exists():
                raise FileNotFoundError(f"缺少模型文件：{path}")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=num_threads,
            sample_rate=RATE,
            feature_dim=80,
            decoding_method=decoding,
            provider=provider,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=rule1,
            rule2_min_trailing_silence=rule2,
            rule3_min_utterance_length=20.0,
        )
        self.stream = self.recognizer.create_stream()
        self._last = ""

    # ---------- 流式接口 ----------

    def accept(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.stream.accept_waveform(RATE, samples)

    def step(self) -> tuple[str, bool]:
        """解一次码，返回 (当前文本, 是否到端点)。"""
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream).strip()
        endpoint = self.recognizer.is_endpoint(self.stream)
        self._last = text
        return text, endpoint

    def reset(self) -> None:
        self.stream = self.recognizer.create_stream()
        self._last = ""

    def finish(self) -> str:
        # 尾巴补 0.5s 静音，让最后几个字也走完
        silence = np.zeros(int(RATE * 0.5), dtype=np.float32)
        self.stream.accept_waveform(RATE, silence)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream).strip()
        self.reset()
        return text

    # ---------- 离线辅助（测试/回放用） ----------

    def transcribe_wav(self, path: str | Path) -> str:
        with wave.open(str(path), "rb") as handle:
            pcm = handle.readframes(handle.getnframes())
            rate = handle.getframerate()
        if rate != RATE:
            pcm = _resample_int16(pcm, rate, RATE)
        self.reset()
        self.accept(pcm)
        text = self.finish()
        return text


def _resample_int16(pcm: bytes, src: int, dst: int) -> bytes:
    """线性插值重采样；语音够用，且不引入依赖。"""
    if src == dst:
        return pcm
    data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_dst = int(round(len(data) * dst / src))
    if n_dst <= 1:
        return b""
    x = np.arange(n_dst, dtype=np.float64) * (src / dst)
    idx = np.floor(x).astype(np.int64)
    frac = (x - idx).astype(np.float32)
    idx = np.clip(idx, 0, len(data) - 1)
    nxt = np.clip(idx + 1, 0, len(data) - 1)
    out = data[idx] * (1.0 - frac) + data[nxt] * frac
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


def _pick(base: Path, *names: str) -> Path:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return base / names[0]


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True
