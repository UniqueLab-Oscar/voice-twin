"""低延迟识别封装：模型只加载一次，草稿与正式共用。

与 ``stages.Asr`` 的区别：
- 直接吃 PCM（不落地成整段文件也行，内部用临时 wav 喂给 faster-whisper）；
- ``cpu_threads`` 按本机核数给足，``beam_size=1``，``without_timestamps``；
- 语言首次自动探测后锁定，避免每句都做语言识别；
- 提供 ``transcribe``（正式）与 ``draft``（草稿）两个入口，草稿只取最短路径。
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

RATE = 16000


def pick_profile(cores: int | None = None, ram_gb: float | None = None) -> str:
    """按硬件选模型档位。"""
    if cores is None:
        cores = os.cpu_count() or 4
    if ram_gb is None:
        ram_gb = 8.0
    if ram_gb < 4 or cores < 4:
        return "tiny"
    if cores < 10:
        return "base"
    return "small"


def _write_wav(path: Path, pcm: bytes, rate: int = RATE) -> None:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


class FastAsr:
    """faster-whisper 的薄封装。"""

    def __init__(
        self,
        model: str = "auto",
        language: str | None = None,
        auto_lang: bool = True,
        threads: int | None = None,
        profile: str = "auto",
    ) -> None:
        self.profile = profile
        self.model_name = model if model != "auto" else (profile if profile != "auto" else pick_profile())
        self.language = language
        self.auto_lang = auto_lang
        self.threads = threads or max(2, min(8, (os.cpu_count() or 4) - 2))
        self.last_language = language or ""
        self._model = None
        self._tmp: tempfile.TemporaryDirectory | None = None

    # ---------- 生命周期 ----------

    def ensure(self) -> float:
        """加载模型；返回加载耗时秒数（已加载则 0）。"""
        if self._model is not None:
            return 0.0
        from faster_whisper import WhisperModel

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        start = time.perf_counter()
        self._model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.threads,
            num_workers=1,
        )
        return time.perf_counter() - start

    def warmup(self) -> float:
        """用 0.4s 静音预热（首次推理会慢很多）。"""
        self.ensure()
        silence = b"\x00\x00" * int(RATE * 0.4)
        start = time.perf_counter()
        self._run(silence, language="en")
        return time.perf_counter() - start

    def close(self) -> None:
        self._model = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    # ---------- 推理 ----------

    def transcribe(self, pcm: bytes, language: str | None = None) -> tuple[str, str, int]:
        """正式识别：返回 (文本, 语言, 毫秒)。"""
        self.ensure()
        start = time.perf_counter()
        text, lang = self._run(pcm, language)
        return text, lang, int((time.perf_counter() - start) * 1000)

    def draft(self, pcm: bytes, language: str | None = None) -> str:
        """草稿识别：同上，但不保证标点完整。"""
        text, _ = self.transcribe(pcm, language)[0], ""
        return text

    def _run(self, pcm: bytes, language: str | None) -> tuple[str, str]:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="vt-asr-")
        path = Path(self._tmp.name) / "chunk.wav"
        _write_wav(path, pcm)

        want = language or self.language
        if self.auto_lang and not self.last_language:
            want = None
        elif self.last_language:
            want = self.last_language

        segments, info = self._model.transcribe(  # type: ignore[union-attr]
            str(path),
            language=want,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            no_speech_threshold=0.6,
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        detected = getattr(info, "language", "") or (want or "")
        if detected and self.auto_lang:
            self.last_language = detected
        return " ".join(parts), detected
