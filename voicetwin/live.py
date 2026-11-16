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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .config import Config
from .seg import RingBuffer, Segmenter, overlap_trim
from .speakers import SpeakerTracker
from .stages import Asr, Interpreter, VoiceTts
from .translate import Translator
from .types import Segment, Translated


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
    partial: bool = False
    lang: str = ""
    asr_ms: int = 0
    tts_ms: int = 0
    total_ms: int = 0


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
        stride: float | None = None,
        on_partial: Callable[[str, str], None] | None = None,
    ) -> None:
        # 实时就是要真模型；mock 只对 transcript 有意义
        cfg.mock = False
        self.cfg = cfg
        self.on_caption = on_caption
        self.on_partial = on_partial
        self.window = window
        # 步长小于窗口 = 重叠切块：每 stride 秒就出一次字，但每次看的是最近 window 秒，
        # 既把延迟压到 stride 量级，又不牺牲识别所需的上下文。
        self.stride = stride or window
        self.rate = rate
        self.tracker = tracker or SpeakerTracker()

        self.asr = Asr(cfg)
        self.interpreter = Interpreter(cfg)
        self.translator = Translator(cfg)
        self.tts = VoiceTts(cfg)

        self._buf = bytearray()
        self._since = 0.0          # 距上次处理又过去了几秒音频
        self._last_text = ""
        self._prompt = ""
        self._consumed = 0.0
        self._index = 0
        self.captions: list[Caption] = []
        self._pending: list = []
        self._tx_pool = None
        self._tmp = Path(tempfile.mkdtemp(prefix="vt-live-"))
        self._tts_broken = False
        # 上一块以连字符截断时先压着，等下一块接上再出字幕
        self._carry_src = ""
        self._carry_dst = ""
        self._carry_t0 = 0.0
        self._carry_speaker = ""

        # ---- 端点检测模式（默认）：静音不出块，说完 ~200ms 就出字 ----
        self.vad_mode = (getattr(cfg, "seg_mode", "vad") or "vad") == "vad"
        self.partial = bool(getattr(cfg, "partial", False))
        self.segmenter = (
            Segmenter(
                min_speech=getattr(cfg, "min_speech", 0.15),
                max_segment=getattr(cfg, "max_segment", 2.0),
                min_silence=getattr(cfg, "min_silence", 0.20),
            )
            if self.vad_mode
            else None
        )
        self.ring = RingBuffer(seconds=8.0, rate=self.rate) if self.vad_mode else None
        self._partial_at = 0.0  # 上次草稿的时间点（秒）
        self._pending_draft = ""

        # ---- 流式识别（sherpa-onnx），比 whisper 快一个数量级 ----
        self.stasr = None
        self._st_partial = ""
        self._seg_t0 = 0.0
        if getattr(cfg, "asr_engine", "whisper") == "stream":
            try:
                from .asr_stream import StreamAsr
                from .paths import model_dir

                model_path = getattr(cfg, "stream_model", "") or str(model_dir())
                self.stasr = StreamAsr(model_path)
                print("[info] 使用 sherpa-onnx 流式识别", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 流式识别不可用，退回 whisper：{exc}", file=sys.stderr)
                self.stasr = None

        # ---- 多语种配音与播放（默认开）----
        self.player = None
        self.edge_tts = None
        self._voice_pool: ThreadPoolExecutor | None = None
        if getattr(cfg, "playback", False) and getattr(cfg, "tts_provider", "edge") == "edge":
            try:
                from .playback import Player, has_output_device
                from .tts_multi import EdgeTts

                if has_output_device() and EdgeTts(cfg.tts_voice if hasattr(cfg, "tts_voice") else "").available():
                    self.player = Player(volume=getattr(cfg, "volume", 1.0))
                    self.edge_tts = EdgeTts(getattr(cfg, "tts_voice", ""))
                    self._voice_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vt-voice")
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 配音不可用，只出字幕：{exc}", file=sys.stderr)
                self.player = None

    # -- 内部 -------------------------------------------------------------
    def _process(self, pcm: bytes, start: float | None = None) -> Caption | None:
        chunk = self._tmp / f"{self._index + 1:04d}.wav"
        _write_wav(chunk, pcm, self.rate)

        speaker = self.tracker.assign(pcm, self.rate)
        t_asr = time.perf_counter()
        segments = self.asr.transcribe(str(chunk), self._prompt)
        asr_ms = int((time.perf_counter() - t_asr) * 1000)
        self._apply_direction()
        if not segments:
            return None
        self._prompt = (self._prompt + " " + segments[-1].text).strip()[-240:]

        base = self._consumed if start is None else start
        for seg in segments:
            seg.start += base
            seg.end += base
            seg.speaker = speaker

        translations = self.interpreter.translate(segments)
        if not translations:
            return None

        text_src = " ".join(s.text for s in segments)
        # 译员层：带上下文与缓存，逐句翻（合并后的整段一起送，保证语气连贯）
        text_dst = self.translator.translate(text_src, self.cfg.src_lang, self.cfg.tgt_lang)

        t0 = base
        if self._carry_src and self._carry_speaker == speaker:
            text_src = self._carry_src + " " + text_src
            text_dst = self._carry_dst + " " + text_dst
            t0 = self._carry_t0
        self._carry_src = self._carry_dst = ""
        self._carry_speaker = ""

        # 重叠窗口会重复上一块的内容：把重复的前缀剪掉，只留新说的部分
        words = text_src.split()
        prev_words = self._last_text.split()
        cut = 0
        for k in range(min(len(words), len(prev_words)), 0, -1):
            if prev_words[-k:] == words[:k]:
                cut = k
                break
        if cut:
            text_src = " ".join(words[cut:])
            text_dst = ""
        self._last_text = " ".join(words)
        if not text_src.strip():
            return None

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
            lang=self.asr.last_language or "",
            asr_ms=asr_ms,
        )

        if self.cfg.tts_engine != "none" and not self._tts_broken:
            try:
                ref_pcm, _ = self.tracker.reference(speaker)
                if ref_pcm and not self.cfg.ref_audio:
                    ref_path = self._tmp / f"ref-{speaker}.wav"
                    _write_wav(ref_path, ref_pcm, self.rate)
                    self.cfg.ref_audio = str(ref_path)
                spoken = Translated(segment=segments[-1], text=text_dst, style=self.cfg.style)
                syn = self.tts.synthesize(spoken, self._index)
                cap.audio = syn.audio_path
            except Exception as exc:  # noqa: BLE001
                # 没有可用克隆引擎时只提示一次，之后安静地只出字幕
                self._tts_broken = True
                print(f"[warn] 换声不可用，之后只出字幕：{exc}", file=sys.stderr)

        self.captions.append(cap)
        if self.player is not None and text_dst.strip():
            self._speak(text_dst)
        if self.on_caption:
            self.on_caption(cap)
        return cap

    def _speak(self, text: str) -> None:
        """后台合成 + 入队播放；不阻塞字幕。"""
        if self.player is None or self.edge_tts is None or self._voice_pool is None:
            return
        self.player.start()
        lang = self.cfg.tgt_lang or "zh"

        def job() -> None:
            try:
                path = self._tmp / f"say-{int(time.time() * 1000)}.wav"
                self.edge_tts.synthesize(text, lang, path)  # type: ignore[union-attr]
                from .tts_multi import read_pcm

                self.player.enqueue(  # type: ignore[union-attr]
                    read_pcm(path), interrupt=getattr(self.cfg, "interrupt", True)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 配音失败：{exc}", file=sys.stderr)

        self._voice_pool.submit(job)

    def _apply_direction(self) -> None:
        """中英自动互译：他说中文就译成英文，说英文就译成中文。

        单段的语言检测会抖（同一段中文偶尔被判成 en），所以要求连续 2 段一致才切换方向，
        否则一段噪声就能把整个会议的方向翻过来。
        """

        if not self.cfg.auto_lang:
            return
        lang = self.asr.last_language or ""
        if not lang:
            return
        src = "zh" if lang.startswith("zh") else "en"
        if src == getattr(self, "_lang_vote", ""):
            self._lang_hits = getattr(self, "_lang_hits", 0) + 1
        else:
            self._lang_vote, self._lang_hits = src, 1
        current = self.cfg.src_lang.split("-")[0].lower()
        if self._lang_hits < 2 or src == current:
            return
        tgt = "en" if src == "zh" else "zh"
        if (self.cfg.src_lang, self.cfg.tgt_lang) != (src, tgt):
            self.cfg.src_lang, self.cfg.tgt_lang = src, tgt
            print(f"[info] 识别到 {lang}，方向 {src} → {tgt}", file=sys.stderr)

    # -- 对外 -------------------------------------------------------------
    def feed(self, pcm: bytes) -> list[Caption]:
        """喂一段 PCM，返回本次产生的正式字幕（草稿走 on_partial 回调）。"""

        if self.stasr is not None:
            return self._feed_stream(pcm)
        if self.vad_mode:
            return self._feed_vad(pcm)

        self._buf.extend(pcm)
        win = int(self.window * self.rate) * 2
        step = int(self.stride * self.rate) * 2
        out: list[Caption] = []
        while len(self._buf) >= max(win, step) and self._since >= self.stride:
            if self.stride >= self.window:
                block = bytes(self._buf[:win])
                self._buf = self._buf[win:]
            else:
                block = bytes(self._buf[-win:])
                self._buf = self._buf[len(self._buf) - win:]
            self._consumed += step / (2 * self.rate)
            self._since = 0.0
            cap = self._process(block)
            if cap:
                out.append(cap)
            if len(self._buf) < win:
                break
        self._since += len(pcm) / (2 * self.rate)
        return out

    def _feed_stream(self, pcm: bytes) -> list[Caption]:
        """流式模式：每 100ms 喂一次，边说边出草稿，端点定稿。"""
        assert self.stasr is not None
        step = len(pcm) / (2 * self.rate)
        was_quiet = not self._st_partial
        if was_quiet:
            self._seg_t0 = max(0.0, self._consumed - 1.0)
        self._consumed += step
        self.stasr.accept(pcm)
        text, endpoint = self.stasr.step()

        out: list[Caption] = []
        if endpoint and text:
            out.append(self._emit_stream(text))
            self.stasr.reset()
            self._st_partial = ""
        elif text and text != self._st_partial:
            self._st_partial = text
            if self.on_partial and self._consumed - self._partial_at >= 0.12:
                self._partial_at = self._consumed
                try:
                    self.on_partial(text, self._detect_lang(text))
                except Exception:  # noqa: BLE001
                    pass
        return out

    @staticmethod
    def _detect_lang(text: str) -> str:
        return "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "en"

    def _emit_stream(self, text: str) -> Caption:
        """把一段流式定稿文本走完整链路：翻译 → 配音 → 字幕。

        翻译放到线程池里做：opus-mt 单句要几秒，同步做会把麦克风读线程堵死、丢音频。
        """
        from .types import Segment

        t0 = self._seg_t0
        t1 = self._consumed
        lang = self._detect_lang(text)
        if self.cfg.auto_lang:
            self.cfg.src_lang = lang
            self.cfg.tgt_lang = "en" if lang == "zh" else "zh"

        cap = Caption(
            index=len(self.captions),
            t0=t0,
            t1=t1,
            speaker="S0",
            source=text,
            translated="",
            lang=lang,
            asr_ms=0.0,
            total_ms=0.0,
        )
        self.captions.append(cap)
        try:
            if self._tx_pool is None:
                from concurrent.futures import ThreadPoolExecutor

                self._tx_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vt-tx")
            fut = self._tx_pool.submit(self._translate_caption, cap, Segment(start=t0, end=t1, text=text), lang)
            self._pending.append(fut)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 翻译线程起不来，退回同步：{exc}")
            self._translate_caption(cap, Segment(start=t0, end=t1, text=text), lang)
        return cap

    def _translate_caption(self, cap: Caption, seg, lang: str) -> None:
        started = time.time()
        try:
            text_dst = self.translator.translate(cap.source, self.cfg.src_lang, self.cfg.tgt_lang)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"[warn] 翻译失败：{exc}")
            text_dst = ""
        cap.translated = text_dst
        cap.total_ms = (time.time() - started) * 1000
        if self.player is not None and text_dst.strip():
            self._speak(text_dst)
        if self.on_caption:
            self.on_caption(cap)

    def drain(self, timeout: float = 30.0) -> None:
        """等翻译队列跑完（停止/收尾时用）。"""
        deadline = time.time() + timeout
        while self._pending:
            fut = self._pending[0]
            left = max(0.0, deadline - time.time())
            try:
                fut.result(timeout=left)
            except Exception:  # noqa: BLE001
                pass
            self._pending.pop(0)

    def _draft(self, pcm: bytes) -> str:
        path = self._tmp / "draft.wav"
        _write_wav(path, pcm, self.rate)
        segments = self.asr.transcribe(str(path), "")
        return " ".join(s.text for s in segments).strip()

    def _feed_vad(self, pcm: bytes) -> list[Caption]:
        """端点检测模式：只对「完整的一句话」跑全链路。"""
        assert self.segmenter is not None and self.ring is not None
        self.ring.push(pcm)
        self._consumed += len(pcm) / (2 * self.rate)
        out: list[Caption] = []
        for span in self.segmenter.feed(pcm):
            cap = self._process(span.pcm, start=span.start)
            if cap:
                out.append(cap)
        if self.partial and self.on_partial and self.segmenter.in_speech:
            if self._consumed - self._partial_at >= 0.35:
                self._partial_at = self._consumed
                window = self.segmenter.partial_window(1.2)
                if len(window) >= int(0.3 * self.rate) * 2:
                    try:
                        text = self._draft(window)
                    except Exception:  # noqa: BLE001
                        text = ""
                    if text and text != self._pending_draft:
                        self._pending_draft = text
                        try:
                            self.on_partial(text, self.asr.last_language or "")
                        except Exception:  # noqa: BLE001
                            pass
        return out

    def flush(self) -> Caption | None:
        """把剩余音频处理掉（说完最后一句时用）。"""

        if self.stasr is not None:
            text = self.stasr.finish()
            if not text:
                return None
            self._st_partial = ""
            return self._emit_stream(text)

        if self.vad_mode and self.segmenter is not None:
            span = self.segmenter.flush()
            if span is None:
                return None
            return self._process(span.pcm, start=span.start)

        if len(self._buf) < int(0.4 * self.rate) * 2:
            return None
        block = bytes(self._buf)
        self._buf.clear()
        self._consumed += len(block) / (2 * self.rate)
        return self._process(block)

    def close(self) -> None:
        self.drain(timeout=120.0)
        if self._tx_pool is not None:
            self._tx_pool.shutdown(wait=False, cancel_futures=True)
            self._tx_pool = None
        if self._voice_pool is not None:
            self._voice_pool.shutdown(wait=False, cancel_futures=True)
            self._voice_pool = None
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:  # noqa: BLE001
                pass
            self.player = None
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


def _srt_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def captions_to_srt(captions: list[Caption]) -> str:
    """把一次会话的字幕写成 SRT（带上说话人前缀）。"""

    blocks: list[str] = []
    for i, cap in enumerate(captions, start=1):
        text = f"{cap.translated}"
        if cap.speaker:
            text = f"[{cap.speaker}] {text}"
        blocks.append(f"{i}\n{_srt_ts(cap.t0)} --> {_srt_ts(cap.t1)}\n{text}\n")
    return "\n".join(blocks)


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
