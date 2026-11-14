"""多语种实时合成：默认走 edge-tts（在线、音色多、首包快），可切本地 piper。

对外只暴露一个函数：``speak(text, lang, out) -> Path``，失败抛异常由上层决定回退。
解码用 miniaudio（自带 mp3 解码器），所以本机没有 ffmpeg 也能跑。
"""

from __future__ import annotations

import asyncio
import tempfile
import wave
from pathlib import Path

RATE = 24000

VOICE_BY_LANG = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ru": "ru-RU-SvetlanaNeural",
}


def voice_for(lang: str, override: str = "") -> str:
    if override:
        return override
    key = (lang or "en").split("-")[0].lower()
    return VOICE_BY_LANG.get(key, "en-US-AriaNeural")


def _decode_mp3(mp3: Path, wav: Path, rate: int = RATE) -> None:
    import miniaudio

    decoded = miniaudio.decode_file(
        str(mp3),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=rate,
    )
    pcm = bytes(memoryview(decoded.samples).cast("B"))
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


async def _edge_save(text: str, voice: str, mp3: Path) -> None:
    import edge_tts

    # 语速略快一点：同传场景听的是信息，不是朗诵
    communicate = edge_tts.Communicate(text, voice, rate="+8%")
    await communicate.save(str(mp3))


def speak_edgesync(text: str, lang: str, out: Path, voice: str = "") -> Path:
    """edge-tts → wav（24k 单声道）。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vt-tts-") as tmp:
        mp3 = Path(tmp) / "speech.mp3"
        asyncio.run(_edge_save(text, voice_for(lang, voice), mp3))
        _decode_mp3(mp3, out)
    return out


def read_pcm(wav_path: Path, rate: int = RATE) -> bytes:
    """读 wav 成 PCM；不是目标采样率时用 miniaudio 重采样。"""
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getframerate() == rate and handle.getnchannels() == 1 and handle.getsampwidth() == 2:
            return handle.readframes(handle.getnframes())
    import miniaudio

    decoded = miniaudio.decode_file(
        str(wav_path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=rate,
    )
    return bytes(memoryview(decoded.samples).cast("B"))


class EdgeTts:
    """薄封装：按语言挑音色，落盘 wav。"""

    name = "edge"

    def __init__(self, voice: str = "") -> None:
        self.voice = voice

    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            import miniaudio  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def synthesize(self, text: str, lang: str, out: Path) -> Path:
        return speak_edgesync(text, lang, out, self.voice)
