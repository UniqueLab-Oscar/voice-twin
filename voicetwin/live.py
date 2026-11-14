"""实时字幕同传：麦克风 → 分块识别 → 译员 → 字幕（可选换声播放）。

和 `stream` 的区别：`stream` 回放文件，`live` 是麦克风实时采，
每一块都立刻出字幕，退出时给一份完整字幕。

音频约定：16 kHz、16bit、单声道、小端 PCM。麦克风与文件回放都统一到这个格式。
"""

from __future__ import annotations

import queue
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .config import Config
from .speakers import SpeakerTracker
from .stages import Asr, Interpreter, VoiceTts
from .types import Segment


@dataclass
class Caption:
    """一条实时字幕。"""

    index: int
    t0: float
    t1: float
    speaker: str
    source: str
    translated: str
    audio: str = ""


def _write_wav(path: Path, pcm: bytes, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


class LiveSession:
    """把连续的 PCM 切成窗口，逐块识别、翻译、可选合成。"""

    def __init__(
        self,
        cfg: Config,
        on_caption: Callable[[Caption], None] | None = None,
        window: float = 1.5,
        rate: int = 16000,
        tracker: SpeakerTracker | None = None,
    ) -> None:
        # 实时就是要真模型；mock 只对 transcript 有意义
        cfg.mock = False
        self.cfg = cfg
        self.on_caption = on_caption
        self.window = window
        self.rate = rate
        self.tracker = tracker or SpeakerTracker()

        self.asr = Asr(cfg)
        self.interpreter = Interpreter(cfg)
        self.tts = VoiceTts(cfg)

        self._buf = bytearray()
        self._prompt = ""
        self._consumed = 0.0
        self._index = 0
        self.captions: list[Caption] = []
        self._tmp = Path(tempfile.mkdtemp(prefix="vt-live-"))
        self._tts_broken = False
        # 上一块以连字符截断时先压着，等下一块接上再出字幕
        self._carry_src = ""
        self._carry_dst = ""
        self._carry_t0 = 0.0
        self._carry_speaker = ""

    # -- 内部 -------------------------------------------------------------
    def _process(self, pcm: bytes) -> Caption | None:
        chunk = self._tmp / f"{self._index + 1:04d}.wav"
        _write_wav(chunk, pcm, self.rate)

        speaker = self.tracker.assign(pcm, self.rate)
        segments = self.asr.transcribe(str(chunk), self._prompt)
        if not segments:
            return None
        self._prompt = (self._prompt + " " + segments[-1].text).strip()[-240:]

        base = self._consumed
        for seg in segments:
            seg.start += base
            seg.end += base
            seg.speaker = speaker

        translations = self.interpreter.translate(segments)
        if not translations:
            return None

        text_src = " ".join(s.text for s in segments)
        text_dst = " ".join(t.text for t in translations)

        t0 = base
        if self._carry_src and self._carry_speaker == speaker:
            text_src = self._carry_src + " " + text_src
            text_dst = self._carry_dst + " " + text_dst
            t0 = self._carry_t0
        self._carry_src = self._carry_dst = ""
        self._carry_speaker = ""

        if text_src.rstrip().endswith("-"):
            # 单词被块边界切断了，压着等下一块，别把半个词推给用户
            self._carry_src = text_src.rstrip().rstrip("-").strip()
            self._carry_dst = text_dst.rstrip().rstrip("-").strip()
            self._carry_t0 = t0
            self._carry_speaker = speaker
            return None

        self._index += 1
        cap = Caption(
            index=self._index,
            t0=t0,
            t1=base + len(pcm) / (2 * self.rate),
            speaker=speaker,
            source=text_src,
            translated=text_dst,
        )

        if self.cfg.tts_engine != "none" and not self._tts_broken:
            try:
                ref_pcm, _ = self.tracker.reference(speaker)
                if ref_pcm and not self.cfg.ref_audio:
                    ref_path = self._tmp / f"ref-{speaker}.wav"
                    _write_wav(ref_path, ref_pcm, self.rate)
                    self.cfg.ref_audio = str(ref_path)
                syn = self.tts.synthesize(translations[0], self._index)
                cap.audio = syn.audio_path
            except Exception as exc:  # noqa: BLE001
                # 没有可用克隆引擎时只提示一次，之后安静地只出字幕
                self._tts_broken = True
                print(f"[warn] 换声不可用，之后只出字幕：{exc}", file=sys.stderr)

        self.captions.append(cap)
        if self.on_caption:
            self.on_caption(cap)
        return cap

    # -- 对外 -------------------------------------------------------------
    def feed(self, pcm: bytes) -> list[Caption]:
        """喂一段 PCM；攒够一个窗口就处理，返回本次产生的字幕。"""

        self._buf.extend(pcm)
        need = int(self.window * self.rate) * 2
        out: list[Caption] = []
        while len(self._buf) >= need:
            block = bytes(self._buf[:need])
            del self._buf[:need]
            self._consumed += need / (2 * self.rate)
            cap = self._process(block)
            if cap:
                out.append(cap)
        return out

    def flush(self) -> Caption | None:
        """把剩余不足一窗的音频也处理掉（说完最后一句时用）。"""

        if len(self._buf) < int(0.4 * self.rate) * 2:
            return None
        block = bytes(self._buf)
        self._buf.clear()
        self._consumed += len(block) / (2 * self.rate)
        return self._process(block)

    def close(self) -> None:
        for p in self._tmp.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            self._tmp.rmdir()
        except OSError:
            pass


def replay_wav(path: str | Path, window: float = 1.5, realtime: bool = True) -> Iterable[bytes]:
    """把 wav 按窗口吐出，模拟麦克风；不是 16k/16bit/单声道会先就地转换。"""

    import audioop

    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        step = int(window * rate)
        while True:
            raw = w.readframes(step)
            if not raw:
                return
            data = raw
            if channels > 1:
                data = audioop.tomono(data, width, 0.5, 0.5)
            if width != 2:
                data = audioop.lin2lin(data, width, 2)
            if rate != 16000:
                data, _ = audioop.ratecv(data, 2, 1, rate, 16000, None)
            yield data
            if realtime:
                time.sleep(window)


def run_mic(session: LiveSession, device: int | None = None, stop: Callable[[], bool] | None = None) -> None:
    """从麦克风读数据喂给 session，Ctrl+C 或 stop() 为真时退出。"""

    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - 取决于本机环境
        raise RuntimeError("实时麦克风需要 sounddevice：pip install sounddevice") from exc

    q: queue.Queue[bytes] = queue.Queue()

    def cb(indata, frames, t, status):  # noqa: ANN001, ARG001
        if status:
            print(f"[warn] 音频状态：{status}", file=sys.stderr)
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=session.rate,
        blocksize=1600,
        device=device,
        dtype="int16",
        channels=1,
        callback=cb,
    ):
        print(f"🎙 正在听（窗口 {session.window:.1f}s），Ctrl+C 停止", file=sys.stderr)
        while not (stop and stop()):
            try:
                pcm = q.get(timeout=0.5)
            except queue.Empty:
                continue
            session.feed(pcm)
        session.flush()


def list_devices() -> str:
    try:
        import sounddevice as sd
    except ImportError:
        return "未安装 sounddevice"
    lines = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            lines.append(f"[{i}] {d['name']}  ({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)")
    return "\n".join(lines) or "没找到可用的录音设备"
