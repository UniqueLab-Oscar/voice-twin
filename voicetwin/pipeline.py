from __future__ import annotations

import time
from pathlib import Path

from .config import Config
from .stages import Asr, Diarizer, Interpreter, Vad, VoiceTts, extract_reference
from .types import RunReport, Synthesis


class Pipeline:
    """VAD → ASR → 说话人分离 → 译员 → 克隆 TTS。

    单阶段失败不拖垮整条链路：记录错误并继续，报告里可见。
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.vad = Vad(cfg)
        self.asr = Asr(cfg)
        self.diarizer = Diarizer(cfg)
        self.interpreter = Interpreter(cfg)
        self.tts = VoiceTts(cfg)

    def run(self, source: str, audio_path: str = "") -> RunReport:
        report = RunReport()
        t0 = time.perf_counter()

        if not self.cfg.mock and audio_path:
            try:
                report.elapsed_ms["vad"] = self._time(lambda: self.vad.detect(audio_path))
            except ImportError as exc:
                # 没装 silero-vad 属于「可选组件缺席」，不是链路故障
                report.elapsed_ms["vad_skip"] = -1
                print(f"[info] VAD 未启用（缺依赖）：{exc}")
            except Exception as exc:  # noqa: BLE001
                report.elapsed_ms["vad_error"] = -1
                print(f"[warn] VAD 失败，跳过：{exc}")

        t_asr = time.perf_counter()
        try:
            report.segments = self.asr.transcribe(source)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"ASR 失败：{exc}") from exc
        report.elapsed_ms["asr"] = (time.perf_counter() - t_asr) * 1000

        t_d = time.perf_counter()
        try:
            report.segments = self.diarizer.assign(report.segments, audio_path)
        except Exception as exc:  # noqa: BLE001
            # 分离失败不该拖垮翻译：退化成单说话人，报告里标出来。
            report.elapsed_ms["diarize_error"] = -1
            print(f"[warn] 说话人分离失败，全部标为 S0：{exc}")
        report.elapsed_ms["diarize"] = (time.perf_counter() - t_d) * 1000

        t_tr = time.perf_counter()
        try:
            report.translations = self.interpreter.translate(report.segments)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"译员阶段失败：{exc}") from exc
        report.elapsed_ms["translate"] = (time.perf_counter() - t_tr) * 1000

        t_tts = time.perf_counter()
        if not self.cfg.mock and audio_path and not self.cfg.ref_audio:
            if self.cfg.tts_engine not in ("sapi", "system"):
                try:
                    ref = extract_reference(
                        audio_path, Path(self.cfg.out_dir) / "reference.wav", self.cfg.ref_seconds
                    )
                    self.cfg.ref_audio = ref
                    print(f"[info] 用源音频前 {self.cfg.ref_seconds:.0f}s 作为说话人参考：{ref}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] 截取参考音频失败（{exc}）；需要换声请显式给 --ref-audio")
        for i, tr in enumerate(report.translations, start=1):
            if self.cfg.tts_engine == "none":
                # 只要字幕、不要配音（Web 演示台的「不合成」）
                report.syntheses.append(Synthesis(translated=tr, audio_path="", voice="none", ok=False))
                continue
            try:
                report.syntheses.append(self.tts.synthesize(tr, i))
            except Exception as exc:  # noqa: BLE001
                # 合成失败也要保留译文行，否则前端会以为整段没识别出来
                print(f"[warn] 第 {i} 句合成失败：{exc}")
                report.syntheses.append(Synthesis(translated=tr, audio_path="", voice="", ok=False))
        report.elapsed_ms["tts"] = (time.perf_counter() - t_tts) * 1000

        report.elapsed_ms["total"] = (time.perf_counter() - t0) * 1000
        return report

    @staticmethod
    def _time(fn) -> float:
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000


def ensure_out_dir(cfg: Config) -> Path:
    p = Path(cfg.out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p
