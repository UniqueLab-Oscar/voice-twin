from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import doctor
from .config import Config
from .pipeline import Pipeline
from .report import dump_run, render_report
from .srt import to_srt


def _fmt_ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config(
        out_dir=args.out,
        mock=args.mock,
        llm_model=args.model,
        style=args.style,
        glossary=args.glossary,
        translator=args.translator,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        auto_lang=args.auto_lang,
        tts_engine=args.tts,
        tts_url=args.tts_url,
        tts_cmd=getattr(args, "tts_cmd", ""),
        ref_audio=args.ref_audio,
        sapi_voice=args.sapi_voice,
        fit_timing=not args.no_fit,
        max_speed=args.max_speed,
        consent=args.consent,
        stamp=not args.no_stamp,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    source = args.transcript if cfg.mock else args.audio
    if not source:
        print("mock 模式需要 --transcript；真实模式需要 --audio", file=sys.stderr)
        return 2
    if not cfg.mock and cfg.tts_engine == "sapi":
        print("[warn] TTS=sapi：Windows 自带音色，只验证「译文→音频」链路，不保留说话人音色")

    report = Pipeline(cfg).run(source, audio_path=args.audio or "")

    print(f"{'#':>3}  {'时间':<6} {'说话人':<8} {'译员版'}")
    for i, syn in enumerate(report.syntheses, start=1):
        seg = syn.translated.segment
        print(f"{i:>3}  {_fmt_ts(seg.start):<6} {seg.speaker:<8} {syn.translated.text}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(report), encoding="utf-8")
    (out / "report.srt").write_text(to_srt(report), encoding="utf-8")
    dump_run(report, out / "run.json")
    print(f"\n报告：{out / 'report.md'}   字幕：{out / 'report.srt'}   中间产物：{out / 'run.json'}")
    print(f"总耗时：{report.elapsed_ms.get('total', 0):.1f} ms")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """同一段音频，直译 vs 译员式重述，并排对比。"""
    results: dict[str, RunReport] = {}
    for style in ("literal", "interpreter"):
        cfg = Config(out_dir=str(Path(args.out) / style), mock=args.mock, style=style, glossary=args.glossary)
        results[style] = Pipeline(cfg).run(args.transcript)

    lit, itp = results["literal"], results["interpreter"]
    print("直译 vs 译员式重述\n")
    for i in range(min(len(lit.translations), len(itp.translations))):
        print(f"[{i + 1}] 原文：{lit.translations[i].segment.text}")
        print(f"    直译：{lit.translations[i].text}")
        print(f"    译员：{itp.translations[i].text}\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# 直译 vs 译员式重述", ""]
    for i in range(min(len(lit.translations), len(itp.translations))):
        lines += [
            f"## {i + 1}",
            f"- 原文：{lit.translations[i].segment.text}",
            f"- 直译：{lit.translations[i].text}",
            f"- 译员：{itp.translations[i].text}",
            "",
        ]
    (out / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"对比报告：{out / 'compare.md'}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"{'项':<10} {'状态':<22} {'结论':<28} 说明")
    print("-" * 96)
    for name, status, ok, hint in doctor.run_all(Config()):
        mark = "✓" if ok else "✗"
        print(f"{mark} {name:<8} {status:<22} {hint:<28}")
    return 0


def cmd_fetch_argos(args: argparse.Namespace) -> int:
    """下载并安装 Argos 离线语言包（一次就好，之后完全断网可用）。"""
    from .stages import prepare_argos_home

    prepare_argos_home()
    import argostranslate.package as pkg

    pkg.update_package_index()
    available = pkg.get_available_packages()
    target = next(
        (p for p in available if p.from_code == args.src and p.to_code == args.tgt), None
    )
    if target is None:
        print(f"没有 {args.src} -> {args.tgt} 的语言包", file=sys.stderr)
        return 1
    path = target.download()
    pkg.install_from_path(path)
    print(f"已安装：{target}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(port=args.port, out_dir=args.out, open_browser=args.open_browser)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .webapp import serve as serve_ui

    serve_ui(port=args.port, open_browser=args.open_browser)
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    """分块同传：不看总耗时，只看「说完多久能出中文」。"""

    from .stages import Asr, Interpreter
    from .stream import StreamRunner, render_srt, render_stream, split_wav

    if not args.audio:
        print("stream 需要 --audio（分块要按时间轴切）", file=sys.stderr)
        return 2

    cfg = Config(
        out_dir=args.out,
        mock=False,
        llm_model=args.model,
        style=args.style,
        glossary=args.glossary,
        translator=args.translator,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        auto_lang=args.auto_lang,
        tts_engine="none",
        api_base=args.api_base,
        api_key=args.api_key,
    )
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunks = split_wav(args.audio, out, args.window)
    if not chunks:
        print("切块结果为空，检查音频文件", file=sys.stderr)
        return 2

    asr = Asr(cfg)
    interpreter = Interpreter(cfg)

    def on_event(ev) -> None:
        text = " / ".join(t.text for t in ev.translations) or "（无语音）"
        lag = ev.lag_ms
        pace = f"落后 {lag:.0f} ms" if lag >= 0 else f"领先 {-lag:.0f} ms"
        print(
            f"[{ev.chunk.start:6.1f}s → {ev.chunk.end:6.1f}s] 延迟 {ev.latency_ms:6.0f} ms  {pace:>12}  {text}",
            flush=True,
        )

    if not cfg.mock:
        warm = asr.warmup()
        print(f"[info] 模型预热 {warm:.0f} ms（不计入延迟统计）")
    print(f"输入：{args.audio}  块数：{len(chunks)}  窗口：{args.window:.1f}s")
    report = StreamRunner(asr.transcribe, interpreter.translate, on_event=on_event).run(chunks)

    (out / "stream.md").write_text(render_stream(report, args.window), encoding="utf-8")
    (out / "stream.srt").write_text(render_srt(report), encoding="utf-8")
    verdict = "可实时" if report.rtf < 1 else "跟不上"
    if report.max_lag_ms > 0:
        pace = f"最大落后 {report.max_lag_ms:.0f} ms"
    else:
        pace = f"全程领先 {abs(report.max_lag_ms):.0f} ms"
    print(f"\nRTF {report.rtf:.2f}（{verdict}）  首块延迟 {report.first_result_ms:.0f} ms  {pace}")
    print(f"报告：{out / 'stream.md'}   字幕：{out / 'stream.srt'}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """实时字幕同传：麦克风直采（或用 --file 回放模拟）。"""

    from .live import LiveSession, replay_wav, run_mic

    cfg = Config(
        out_dir=args.out,
        mock=False,
        llm_model=args.model,
        style=args.style,
        glossary=args.glossary,
        translator=args.translator,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        auto_lang=args.auto_lang,
        tts_engine=args.tts if not args.no_voice else "none",
        tts_url=args.tts_url,
        tts_cmd=getattr(args, "tts_cmd", ""),
        api_base=args.api_base,
        api_key=args.api_key,
        seg_mode=getattr(args, "seg", "vad"),
        max_segment=getattr(args, "max_segment", 2.0),
        min_silence=getattr(args, "min_silence", 0.20),
        partial=getattr(args, "partial", False),
        quality=getattr(args, "quality", "fast"),
        playback=not (getattr(args, "no_voice", False) or getattr(args, "no_play", False)),
        volume=getattr(args, "volume", 1.0),
        tts_voice=getattr(args, "voice", ""),
        tts_provider=getattr(args, "tts_provider", "edge"),
        asr_engine=getattr(args, "asr_engine", "whisper"),
        stream_model=getattr(args, "stream_model", ""),
    )

    def show(cap) -> None:
        print(f"[{_fmt_ts(cap.t0)}] {cap.speaker}  {cap.source}")
        cost = f"  (识别 {cap.asr_ms:.0f}ms"
        if cap.translated:
            cost += f" · 译完 {cap.total_ms:.0f}ms"
        cost += ")"
        print(f"          → {cap.translated}{cost}" + (f"  (♪ {cap.audio})" if cap.audio else ""), flush=True)

    def partial(text: str, lang: str) -> None:
        print(f"[···] {text}", flush=True)

    session = LiveSession(cfg, on_caption=show, window=args.window, on_partial=partial)
    print(f"引擎：asr={cfg.asr_model} translator={cfg.translator} tts={cfg.tts_engine} seg={cfg.seg_mode}")
    try:
        if args.file:
            for pcm in replay_wav(args.file, args.window, realtime=not args.fast):
                session.feed(pcm)
            session.flush()
        else:
            run_mic(session, device=args.device)
    except KeyboardInterrupt:
        print("\n[info] 停止")
    finally:
        session.flush()
        print(f"\n共 {len(session.captions)} 句 · 说话人：{session.tracker.summary()}")
        session.close()
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """打开图形应用（tkinter）。"""

    from .app import launch

    cfg = Config(
        out_dir=args.out,
        mock=False,
        llm_model=args.model,
        style=args.style,
        translator=args.translator,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        auto_lang=args.auto_lang,
        tts_engine=args.tts,
        tts_url=args.tts_url,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    launch(cfg, window=args.window, device=args.device)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，中文/日文会变成乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    parser = argparse.ArgumentParser(prog="voicetwin", description="保留说话人音色的同声传译")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="out", help="输出目录")
    common.add_argument("--mock", dest="mock", action="store_true", default=True)
    common.add_argument("--no-mock", dest="mock", action="store_false")
    common.add_argument("--model", default="qwen2.5:7b", help="译员 LLM 模型名")
    common.add_argument("--style", choices=["interpreter", "literal"], default="interpreter")
    common.add_argument("--glossary", default="", help="术语表文件（.json 或 每行 源=译 的 .txt）")
    common.add_argument(
        "--translator",
        choices=["ollama", "openai", "argos", "marian", "passthrough"],
        default="ollama",
    )
    common.add_argument("--auto-lang", action="store_true", help="中英自动互译：识别到中文就译英文，反之译中文")
    common.add_argument("--asr-model", default="small", help="faster-whisper 模型：tiny/base/small/medium/large-v3")
    common.add_argument("--asr-device", default="auto", help="auto | cpu | cuda")
    common.add_argument("--tts", default="cosyvoice2", choices=["cosyvoice2", "gpt-sovits", "http", "sapi", "system"],
                        help="sapi = Windows 自带 TTS（不换声，仅验证链路）；cosyvoice2/gpt-sovits/http 需要 --tts-url")
    common.add_argument("--tts-url", default="", help="克隆 TTS 的 HTTP 端点，如 http://127.0.0.1:9880/synthesize")
    common.add_argument("--tts-cmd", default="",
                        help="外部合成命令模板，支持 {text} {ref} {out}，如 \"python synth.py --text {text} --ref {ref} --out {out}\"")
    common.add_argument("--ref-audio", default="", help="说话人参考音频（wav）；留空则自动截源音频前 8 秒")
    common.add_argument("--sapi-voice", default="", help="sapi 音色名，如 zh-CN 女声")
    common.add_argument("--no-fit", dest="no_fit", action="store_true",
                        help="不做时间轴对齐（默认会把每句对齐回原句时长）")
    common.add_argument("--max-speed", type=float, default=1.3, help="时间轴对齐允许的最大加速倍率（默认 1.3）")
    common.add_argument("--consent", choices=["self", "granted", "public-domain"], default="self",
                        help="音色授权来源，会写进音频的 vtwn 溯源块")
    common.add_argument("--no-stamp", dest="no_stamp", action="store_true", help="不写入来源/同意声明块")
    common.add_argument("--api-base", default="https://api.openai.com/v1")
    common.add_argument("--api-key", default=os.environ.get("VOICETWIN_API_KEY", ""))

    run = sub.add_parser("run", parents=[common], help="跑一遍完整链路")
    run.add_argument("--transcript", help="mock/离线模式：TSV 字幕文件")
    run.add_argument("--audio", help="真实音频文件（wav/mp3）")
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", parents=[common], help="直译 vs 译员式重述对比")
    cmp.add_argument("--transcript", required=True)
    cmp.set_defaults(func=cmd_compare)

    liv = sub.add_parser("live", parents=[common], help="实时字幕同传（麦克风直采，或 --file 回放）")
    liv.add_argument("--file", default="", help="用 wav 回放模拟麦克风（16k/16bit/单声道）")
    liv.add_argument("--fast", action="store_true", help="回放时全速，不按真实时间等待")
    liv.add_argument("--window", type=float, default=2.5, help="每块窗口秒数（默认 1.5）")
    liv.add_argument("--device", type=int, default=None, help="录音设备编号，见 --list-devices")
    liv.add_argument("--no-voice", action="store_true", help="只出字幕，不合成声音")
    liv.add_argument("--seg", choices=["vad", "fixed"], default="vad", help="分段方式：vad=端点检测（快）")
    liv.add_argument("--max-segment", type=float, default=2.0, help="单句最长秒数，超了强切")
    liv.add_argument("--min-silence", type=float, default=0.20, help="说完多久算一句结束")
    liv.add_argument("--partial", action="store_true", help="边说边出草稿（更跟手，费一点 CPU）")
    liv.add_argument("--quality", choices=["fast", "best"], default="fast", help="翻译档位")
    liv.add_argument("--no-play", action="store_true", help="只出字幕，不播放译音")
    liv.add_argument("--voice", default="", help="edge-tts 音色名，如 zh-CN-XiaoxiaoNeural")
    liv.add_argument("--tts-provider", choices=["edge", "http", "sapi"], default="edge", help="配音引擎")
    liv.add_argument("--volume", type=float, default=1.0, help="播放音量 0–2")
    liv.add_argument(
        "--asr-engine",
        choices=["whisper", "stream"],
        default="whisper",
        help="stream=sherpa-onnx 流式识别（快，边说边出字）",
    )
    liv.add_argument("--stream-model", default="", help="流式模型目录，默认 models/st-zh-en")
    liv.set_defaults(func=cmd_live)

    app = sub.add_parser("app", parents=[common], help="打开图形界面（实时字幕 + 换声 + 说话人）")
    app.add_argument("--window", type=float, default=2.5, help="每块窗口秒数")
    app.add_argument("--device", type=int, default=None, help="录音设备编号")
    app.set_defaults(func=cmd_app)

    stm = sub.add_parser("stream", parents=[common], help="分块同传：边切边译，量化延迟")
    stm.add_argument("--audio", required=True, help="要模拟同传的音频文件（wav）")
    stm.add_argument("--window", type=float, default=3.0, help="分块窗口秒数（默认 3.0）")
    stm.set_defaults(func=cmd_stream)

    doc = sub.add_parser("doctor", help="检查本机环境（Python/ffmpeg/Ollama/GPU）")
    doc.set_defaults(func=cmd_doctor)

    srv = sub.add_parser("serve", help="启动本地 Web 演示台（只监听 127.0.0.1）")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--out", default="out/web", help="输出目录")
    srv.add_argument("--no-open", dest="open_browser", action="store_false", default=True)
    srv.set_defaults(func=cmd_serve)

    ui = sub.add_parser("ui", help="打开实时同传控制台（深色界面 + 流式识别）")
    ui.add_argument("--port", type=int, default=8766)
    ui.add_argument("--no-open", dest="open_browser", action="store_false", default=True)
    ui.set_defaults(func=cmd_ui)

    fa = sub.add_parser("fetch-argos", help="下载 Argos 离线翻译语言包")
    fa.add_argument("--src", default="en")
    fa.add_argument("--tgt", default="zh")
    fa.set_defaults(func=cmd_fetch_argos)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
