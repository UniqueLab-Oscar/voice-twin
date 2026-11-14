"""用 edge-tts 生成多语种测试音频（16k 单声道 wav），给 live/stream 做基准。"""

from __future__ import annotations

import argparse
import asyncio
import struct
import wave
from pathlib import Path

VOICES = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
}

SAMPLES = {
    "zh": "今天下午三点开会，请大家提前十分钟到会议室。这个季度的目标是提升响应速度。",
    "en": "We are going to speed up the pipeline today. The target is under one second.",
    "ja": "今日は会議が三時から始まります。資料を準備してください。",
}


def mp3_to_wav(mp3: Path, wav: Path, rate: int = 16000) -> float:
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
    return len(pcm) / (2 * rate)


async def synth(text: str, voice: str, out: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="zh", choices=sorted(SAMPLES))
    parser.add_argument("--text", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    lang = args.lang
    text = args.text or SAMPLES[lang]
    target = Path(args.out) if args.out else Path("demo") / f"sample_{lang}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    mp3 = target.with_suffix(".mp3")

    asyncio.run(synth(text, VOICES[lang], mp3))
    seconds = mp3_to_wav(mp3, target)
    print(f"{target}  {seconds:.1f}s  {target.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
