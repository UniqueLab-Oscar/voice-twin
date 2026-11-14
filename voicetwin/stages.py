"""VoiceTwin 的五个可插拔阶段。

每个阶段都有 mock 实现（零依赖，用于验证链路与控制流）和 real 实现
（按需懒加载，未安装依赖时给出可读的报错）。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

from .config import Config
from .provenance import assert_wav_playable, build_meta, stamp
from .timing import fit_to_timeline, is_wav
from .types import Segment, Synthesis, Translated

INTERPRETER_PROMPT = """你是一名资深同声传译员。听者不懂{src}，只能听你的{tgt}译文。
规则：
1. 以「让听者听懂」为第一目标：可压缩冗余、合并短句、补必要背景。
2. 保留说话人的语气、幽默与强调，不要书面化。
3. 术语统一；不确定的专有名词保留原文读音。
4. 只输出译文本身，不要解释、不要加引号。"""


# --------------------------------------------------------------------------
# 音频工具（仅标准库）
# --------------------------------------------------------------------------
def read_wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def write_tone(path: str, seconds: float, freq: float = 440.0, sr: int = 16000) -> None:
    """mock 合成：写一段正弦波，证明产物落盘与命名规则。"""
    n = max(1, int(seconds * sr))
    amp = 6000
    buf = bytearray()
    for i in range(n):
        v = int(amp * math.sin(2 * math.pi * freq * i / sr))
        buf += int(v).to_bytes(2, "little", signed=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(buf))


def extract_reference(src: str, dst: Path, seconds: float = 8.0) -> str:
    """从源音频截前 N 秒作为说话人参考（自举：不预先准备参考音频也能换声）。

    只处理 wav（标准库能读）；其它格式请用 --ref-audio 指定转换好的 wav。
    """
    if not str(src).lower().endswith(".wav"):
        raise RuntimeError("自动截取参考音频只支持 wav；其它格式请用 --ref-audio 指定")
    with wave.open(str(src), "rb") as r:
        params = r.getparams()
        frames = r.readframes(int(params.framerate * seconds))
    if not frames:
        raise RuntimeError("源音频为空，无法生成参考音频")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(frames)
    return str(dst)


def load_transcript(path: str) -> list[Segment]:
    """读取 TSV：start<TAB>end<TAB>speaker<TAB>text。"""
    segs: list[Segment] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) < 4:
            continue
        segs.append(Segment(float(parts[0]), float(parts[1]), "\t".join(parts[3:]), parts[2]))
    return segs


def prepare_argos_home() -> None:
    """把 Argos 的数据/配置/缓存目录指到项目内。

    Argos 默认写 `~/.local/share`，在受限进程里会 `PermissionError`；
    环境变量必须在 `import argostranslate` 之前生效，所以单独抽出来。
    """
    root = Path(__file__).resolve().parent.parent / ".argos"
    os.environ.setdefault("XDG_DATA_HOME", str(root / "data"))
    os.environ.setdefault("XDG_CONFIG_HOME", str(root / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    os.environ.setdefault("STANZA_RESOURCES_DIR", str(root / "stanza_resources"))
    # stanza 每次初始化都会去 raw.githubusercontent.com 拉模型索引，国内会超时；换 jsDelivr 镜像。
    os.environ.setdefault(
        "STANZA_RESOURCES_URL",
        "https://cdn.jsdelivr.net/gh/stanfordnlp/stanza-resources@main",
    )
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    (root / "data").mkdir(parents=True, exist_ok=True)
    _mirror_hf_requests()
    _patch_stanza_download()


def _patch_stanza_download() -> None:
    """stanza 的 download_file 强依赖 content-length，jsDelivr 这类分块响应没有该头。

    换成「流式写盘、不读 content-length」的实现，镜像才可用。
    """
    try:
        import requests
        from stanza.resources import common as stanza_common
    except ImportError:
        return
    if getattr(stanza_common, "_voicetwin_patched", False):
        return

    def download_file(url, path, proxies, raise_for_status=False):  # type: ignore[no-untyped-def]
        r = requests.get(url, stream=True, proxies=proxies, timeout=120)
        if raise_for_status:
            r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)
                    f.flush()
        return r.status_code

    stanza_common.download_file = download_file
    stanza_common._voicetwin_patched = True  # type: ignore[attr-defined]


def _mirror_hf_requests() -> None:
    """stanza 自己拼 `https://huggingface.co/...` 去 requests.get，不认 HF_ENDPOINT。

    国内直连 huggingface.co 会超时，这里在进程内把该域名的请求改写到镜像。
    只影响 `huggingface.co` 前缀，其它请求原样放行。
    """
    import requests

    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if not endpoint or "huggingface.co" in endpoint:
        return
    if getattr(requests, "_voicetwin_mirrored", False):
        return

    original_get = requests.get

    def get(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(url, str) and "://huggingface.co" in url:
            url = endpoint + url.split("huggingface.co", 1)[1]
        return original_get(url, *args, **kwargs)

    requests.get = get  # type: ignore[assignment]
    requests._voicetwin_mirrored = True  # type: ignore[attr-defined]


def load_glossary(path: str) -> dict[str, str]:
    """术语表：.json 为 {源: 译}；.txt 每行 `源=译`，忽略空行与 # 注释。"""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    if p.suffix.lower() == ".json":
        return {str(k): str(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    table: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        src, tgt = line.split("=", 1)
        table[src.strip()] = tgt.strip()
    return table


# --------------------------------------------------------------------------
# 1. VAD
# --------------------------------------------------------------------------
class Vad:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def detect(self, audio_path: str) -> list[tuple[float, float]]:
        if self.cfg.mock or not audio_path:
            return []
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

        model = load_silero_vad(onnx=True)
        wav = read_audio(audio_path, sampling_rate=16000)
        ts = get_speech_timestamps(wav, model, return_seconds=True)
        return [(float(t["start"]), float(t["end"])) for t in ts]


# --------------------------------------------------------------------------
# 2. ASR
# --------------------------------------------------------------------------
class Asr:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # 分块同传会反复调用 transcribe；模型只在第一次加载，避免每块都重新载入
        self._model = None

    def transcribe(self, source: str, prompt: str = "") -> list[Segment]:
        """`prompt` 是上一块已识别的尾部文本，用来让模型接上下文，少切句子。"""

        if source.lower().endswith((".tsv", ".txt")):
            return load_transcript(source)
        if self.cfg.mock:
            raise RuntimeError("mock 模式只能读 transcript；识别真实音频请设 mock=False")
        # 国内网络直连 huggingface.co 基本不通；默认走镜像，用户可用环境变量覆盖。
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(self.cfg.asr_model, device=self.cfg.asr_device, compute_type="int8")
        segs, _info = self._model.transcribe(
            source,
            language=self.cfg.src_lang,
            vad_filter=True,
            initial_prompt=prompt or None,
        )
        out: list[Segment] = []
        for s in segs:
            text = s.text.strip()
            # 静音块常被模型脑补成「Thank you.」之类；no_speech_prob 高的直接丢掉
            if not text or float(getattr(s, "no_speech_prob", 0.0)) > 0.6:
                continue
            out.append(Segment(float(s.start), float(s.end), text))
        return out

    def warmup(self) -> float:
        """加载模型并空跑 0.3s 静音，把首次加载成本挡在延迟统计之外。

        返回预热耗时（毫秒）；mock 模式下直接返回 0。
        """

        if self.cfg.mock:
            return 0.0
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silence.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 4800)
            self.transcribe(str(path))
        return (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------
# 3. 说话人分离
# --------------------------------------------------------------------------
class Diarizer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def assign(self, segments: list[Segment], audio_path: str = "") -> list[Segment]:
        if self.cfg.mock or not audio_path:
            return segments
        from pyannote.audio import Pipeline as PyannotePipeline

        pipe = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        ann = pipe(audio_path)
        for seg in segments:
            best, best_overlap = seg.speaker, 0.0
            for turn, _track, spk in ann.itertracks(yield_label=True):
                overlap = max(0.0, min(seg.end, turn.end) - max(seg.start, turn.start))
                if overlap > best_overlap:
                    best, best_overlap = spk, overlap
            seg.speaker = best
            seg.ref_audio = audio_path
        return segments


# --------------------------------------------------------------------------
# 4. 译员（不是翻译机）
# --------------------------------------------------------------------------
class Interpreter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        demo_dir = Path(__file__).resolve().parent.parent / "demo"
        self.pairs = self._load(demo_dir / "mock_pairs.json")
        self.literal_pairs = self._load(demo_dir / "mock_pairs_literal.json")
        self.glossary = load_glossary(cfg.glossary)

    @staticmethod
    def _load(path: Path) -> dict[str, str]:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def translate(self, segments: list[Segment]) -> list[Translated]:
        return [Translated(seg, self._one(seg.text), self.cfg.style) for seg in segments]

    def _one(self, text: str) -> str:
        if self.cfg.mock:
            table = self.literal_pairs if self.cfg.style == "literal" else self.pairs
            return self._apply_glossary(table.get(text) or self._compress(text))
        if self.cfg.translator == "passthrough":
            return text
        if self.cfg.translator == "argos":
            return self._argos(text)
        if self.cfg.translator == "openai":
            return self._openai(text)
        return self._ollama(text)

    def _argos(self, text: str) -> str:
        """离线 NMT（Argos/CTranslate2）：没有 LLM 也能出中文，用于演示与兜底。

        质量明显弱于 LLM，只适合「先跑通」；正式使用请走 ollama / openai。
        """
        prepare_argos_home()
        from argostranslate import translate as argos_translate

        langs = argos_translate.get_installed_languages()
        src = next((lang for lang in langs if lang.code == self.cfg.src_lang), None)
        tgt = next((lang for lang in langs if lang.code == self.cfg.tgt_lang), None)
        if src is None or tgt is None:
            raise RuntimeError(
                f"Argos 缺少 {self.cfg.src_lang}->{self.cfg.tgt_lang} 语言包，先运行："
                "python -m voicetwin fetch-argos"
            )
        return src.get_translation(tgt).translate(text).strip()

    def _apply_glossary(self, text: str) -> str:
        for src, tgt in self.glossary.items():
            text = text.replace(src, tgt)
        return text

    @staticmethod
    def _compress(text: str) -> str:
        for filler in ("you know,", "I mean,", "um,", "uh,", "like,"):
            text = text.replace(filler, "")
        return " ".join(text.split())

    def _system(self) -> str:
        prompt = INTERPRETER_PROMPT.format(src=self.cfg.src_lang, tgt=self.cfg.tgt_lang)
        if self.glossary:
            prompt += "\n术语表（必须遵守）：\n" + "\n".join(
                f"- {src} → {tgt}" for src, tgt in self.glossary.items()
            )
        return prompt

    def _ollama(self, text: str) -> str:
        req = urllib.request.Request(
            self.cfg.ollama_url.rstrip("/") + "/api/chat",
            data=json.dumps(
                {
                    "model": self.cfg.llm_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": self._system()},
                        {"role": "user", "content": text},
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))["message"]["content"].strip()
        except (urllib.error.URLError, KeyError) as exc:  # pragma: no cover - 依赖外部服务
            raise RuntimeError(
                f"译员 LLM 不可用（{self.cfg.ollama_url}, model={self.cfg.llm_model}）：{exc}"
            ) from exc

    def _openai(self, text: str) -> str:
        if not self.cfg.api_key:
            raise RuntimeError("openai 模式需要 API key：--api-key 或环境变量 VOICETWIN_API_KEY")
        req = urllib.request.Request(
            self.cfg.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(
                {
                    "model": self.cfg.llm_model,
                    "temperature": 0.3,
                    "messages": [
                        {"role": "system", "content": self._system()},
                        {"role": "user", "content": text},
                    ],
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, IndexError) as exc:  # pragma: no cover
            raise RuntimeError(f"译员 API 不可用（{self.cfg.api_base}）：{exc}") from exc


# --------------------------------------------------------------------------
# 5. 音色克隆 TTS
# --------------------------------------------------------------------------
class VoiceTts:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def synthesize(self, translated: Translated, idx: int) -> Synthesis:
        out = Path(self.cfg.out_dir) / f"{idx:03d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        voice = translated.segment.speaker or "S0"

        if self.cfg.mock:
            write_tone(str(out), 0.4, 440.0 + idx * 40)
        elif self.cfg.tts_engine == "sapi":
            self._sapi(translated.text, str(out))
        else:
            self._produce_cloned(translated, out)

        self._finalize(out, translated)
        return Synthesis(translated, str(out), voice)

    def _produce_cloned(self, translated: Translated, out: Path) -> None:
        ref = self._reference(translated)

        if self.cfg.tts_url:
            self._http(translated.text, ref, str(out))
            return

        if not self.cfg.tts_cmd:
            raise RuntimeError(
                "没有可用的克隆 TTS 引擎：\n"
                "  - 接本地 HTTP 克隆服务：--tts-url http://127.0.0.1:9880/synthesize\n"
                "  - 或给一条外部命令模板：--tts-cmd \"python synth.py --text {text} --ref {ref} --out {out}\"\n"
                "  - 只想看链路：--tts sapi（不换声）或 --mock（占位音）"
            )
        cmd = self.cfg.tts_cmd.format(
            text=translated.text.replace('"', "'"),
            ref=ref,
            out=str(out),
        )
        subprocess.run(cmd, shell=True, check=True)

    def _finalize(self, out: Path, translated: Translated) -> None:
        """后处理：对齐回原时间轴 + 写入来源/同意声明。

        两步都是「补一刀就多一层价值」的独立能力，失败只告警，不影响主链路。
        """
        if not out.exists() or out.stat().st_size == 0:
            return
        # 外部引擎可能返回 mp3/flac；只有真正能解析的 wav 才做后处理
        if not is_wav(out):
            print("[info] 合成结果不是 wav，跳过时间轴对齐与溯源块")
            return

        seg = translated.segment
        if self.cfg.fit_timing:
            target = max(0.05, seg.end - seg.start)
            try:
                fitted = out.with_name(out.stem + ".fit.wav")
                fit_to_timeline(out, fitted, target, self.cfg.max_speed)
                os.replace(fitted, out)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 时间轴对齐失败，保留原音频：{exc}")

        if self.cfg.stamp:
            try:
                meta = build_meta(
                    engine=self.cfg.tts_engine if not self.cfg.mock else "mock",
                    reference=self.cfg.ref_audio or None,
                    consent=self.cfg.consent,
                )
                stamp(out, meta)
                assert_wav_playable(out)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 溯源块写入失败（音频仍可用）：{exc}")

    def _reference(self, translated: Translated) -> str:
        """说话人参考音频：优先逐句分离结果，其次全局参考，再次源音频。"""
        return translated.segment.ref_audio or self.cfg.ref_audio

    def _http(self, text: str, ref: str, out: str) -> None:
        """调用外部克隆 TTS。契约：POST JSON -> 音频字节（wav）。

        请求：{"text": "...", "reference_audio": "/abs/ref.wav", "language": "zh"}
        响应：Content-Type 为 audio/*，body 为 wav 字节。
        """
        import urllib.request

        payload = json.dumps(
            {"text": text, "reference_audio": ref, "language": self.cfg.tgt_lang},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.tts_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        if not data:
            raise RuntimeError(f"克隆 TTS 返回空音频：{self.cfg.tts_url}")
        Path(out).write_bytes(data)

    def _sapi(self, text: str, out: str) -> None:
        """Windows 自带 TTS（System.Speech）：离线、零额外依赖。

        它**不换声**，只用来验证「译文能变成音频并按句落盘」这一段链路；
        真正保留音色的是 cosyvoice2 / gpt-sovits 引擎。
        """
        quote = lambda s: s.replace("'", "''")  # noqa: E731
        # System.Speech 对相对路径（尤其带 / 的）会先建出 0 字节文件再失败，必须用绝对路径。
        abs_out = quote(str(Path(out).resolve()))
        pick = f"$s.SelectVoice('{quote(self.cfg.sapi_voice)}');" if self.cfg.sapi_voice else ""
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            f"{pick}"
            f"$s.SetOutputToWaveFile('{abs_out}');"
            f"$s.Speak('{quote(text)}');"
            "$s.Dispose();"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
        )
        # Speak 失败时 PowerShell 仍返回 0，只会留下 0 字节文件：必须自己验产物。
        if not Path(out).exists() or Path(out).stat().st_size == 0:
            raise RuntimeError(
                "SAPI 合成失败：未产出音频。常见原因是系统未安装语音，"
                "或当前进程的安全策略不允许调用语音引擎。"
            )
