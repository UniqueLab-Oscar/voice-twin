"""一个窗口的应用：实时字幕同传 + 换声 + 说话人标签。

只用 tkinter（Python 自带），不额外装 GUI 依赖。麦克风采集在子线程，
字幕通过队列回到界面线程，避免 Tk 跨线程崩溃。
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

from .config import Config
from .live import Caption, LiveSession, captions_to_srt, list_devices, run_mic

SPEAKER_COLORS = ["#5eead4", "#f9a8d4", "#fcd34d", "#93c5fd", "#c4b5fd", "#fca5a5"]


class App:
    def __init__(self, cfg: Config, window: float = 1.5, device: int | None = None) -> None:
        self.cfg = cfg
        self.window = window
        self.device = device
        self.session: LiveSession | None = None
        self.worker: threading.Thread | None = None
        self.events: "queue.Queue[Caption]" = queue.Queue()
        self.stop_flag = threading.Event()
        self.running = False

        self.root = tk.Tk()
        self.root.title("VoiceTwin · 实时字幕同传")
        self.root.geometry("980x620")
        self.root.configure(bg="#0f172a")

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._pump)

    # -- 界面 -------------------------------------------------------------
    def _build(self) -> None:
        top = tk.Frame(self.root, bg="#0f172a")
        top.pack(fill="x", padx=14, pady=10)

        self.btn = tk.Button(
            top, text="▶ 开始监听", command=self.toggle, bg="#2563eb", fg="white",
            activebackground="#1d4ed8", activeforeground="white", relief="flat", padx=16, pady=6,
        )
        self.btn.pack(side="left")

        self.voice = tk.BooleanVar(value=self.cfg.tts_engine != "none")
        tk.Checkbutton(
            top, text="替换成对方的声音", variable=self.voice, command=self._toggle_voice,
            bg="#0f172a", fg="#cbd5e1", selectcolor="#1e293b", activebackground="#0f172a",
            activeforeground="#cbd5e1",
        ).pack(side="left", padx=12)

        self.status = tk.Label(top, text="未开始", fg="#94a3b8", bg="#0f172a")
        self.status.pack(side="left", padx=8)

        self.devices = ttk.Combobox(top, width=34, state="readonly")
        self.devices.pack(side="right")
        names = [d for d in list_devices().splitlines() if d.strip()]
        self.devices["values"] = names or ["（未找到录音设备）"]
        if names:
            # 默认挑真麦克风，别用「立体声混音 / 扬声器」
            want = ("麦克风", "Microphone", "阵列", "Headset")
            bad = ("混音", "扬声器", "Stereo Mix", "Speakers")
            pick = next(
                (i for i, n in enumerate(names) if any(w in n for w in want) and not any(b in n for b in bad)),
                0,
            )
            self.devices.current(pick)

        body = tk.Frame(self.root, bg="#111827")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 8))

        self.text = tk.Text(
            body, bg="#111827", fg="#e5e7eb", insertbackground="#e5e7eb",
            wrap="word", relief="flat", font=("Microsoft YaHei UI", 12), padx=12, pady=10,
        )
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("src", foreground="#94a3b8")
        self.text.tag_configure("dst", foreground="#f8fafc")
        self.text.tag_configure("ts", foreground="#64748b")
        for i, color in enumerate(SPEAKER_COLORS):
            self.text.tag_configure(f"spk{i}", foreground=color)
        self.text.configure(state="disabled")

        self.foot = tk.Label(
            self.root, text="音频只在本机处理，不上传任何服务器", fg="#475569", bg="#0f172a", anchor="w",
        )
        self.foot.pack(fill="x", padx=14, pady=(0, 8))

    # -- 控制 -------------------------------------------------------------
    def toggle(self) -> None:
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        cfg = Config(
            out_dir=self.cfg.out_dir,
            mock=False,
            asr_model=self.cfg.asr_model,
            asr_device=self.cfg.asr_device,
            translator=self.cfg.translator,
            style=self.cfg.style,
            tts_engine=self.cfg.tts_engine if self.voice.get() else "none",
            tts_url=self.cfg.tts_url,
            api_base=self.cfg.api_base,
            api_key=self.cfg.api_key,
        )
        self.session = LiveSession(cfg, on_caption=self.events.put, window=self.window)
        self.stop_flag.clear()
        self.running = True
        self.btn.configure(text="■ 停止", bg="#dc2626")
        self.status.configure(text="正在听…", fg="#22c55e")

        device = None
        try:
            name = self.devices.get()
            if name.startswith("["):
                device = int(name[1:name.index("]")])
        except (ValueError, IndexError):
            device = None

        def work() -> None:
            try:
                run_mic(self.session, device=device, stop=self.stop_flag.is_set)
            except Exception as exc:  # noqa: BLE001
                self.events.put(Caption(0, 0, 0, "err", "麦克风错误", str(exc)))
            finally:
                self.root.after(0, self._stopped)

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_flag.set()

    def _stopped(self) -> None:
        self.running = False
        self.btn.configure(text="▶ 开始监听", bg="#2563eb")
        self.status.configure(text="已停止", fg="#94a3b8")
        if self.session:
            saved = self._save_srt()
            note = f" · 已存 {saved}" if saved else ""
            self.status.configure(
                text=f"已停止 · 共 {len(self.session.captions)} 句 · {self.session.tracker.summary()}{note}",
                fg="#94a3b8",
            )

    def _save_srt(self) -> str:
        """停止时把本次字幕落盘，方便复盘或挂播放器。"""

        if not (self.session and self.session.captions):
            return ""
        try:
            from pathlib import Path
            import time as _time

            out = Path(self.cfg.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"live-{_time.strftime('%Y%m%d-%H%M%S')}.srt"
            path.write_text(captions_to_srt(self.session.captions), encoding="utf-8")
            self._write(f"\n      💾 字幕已保存：{path}\n", "ts")
            return str(path)
        except Exception as exc:  # noqa: BLE001
            self._write(f"\n      ⚠ 字幕保存失败：{exc}\n", "ts")
            return ""

    def _toggle_voice(self) -> None:
        self.status.configure(text="换声：" + ("开" if self.voice.get() else "关"), fg="#94a3b8")

    # -- 刷新 -------------------------------------------------------------
    def _pump(self) -> None:
        try:
            while True:
                cap = self.events.get_nowait()
                self._append(cap)
        except queue.Empty:
            pass
        self.root.after(120, self._pump)

    def _append(self, cap: Caption) -> None:
        if cap.speaker == "err":
            self._write(f"\n⚠ {cap.source}：{cap.translated}\n", "dst")
            return
        idx = int(cap.speaker[1:]) if cap.speaker[1:].isdigit() else 0
        self._write(f"\n{_ts(cap.t0)} ", "ts")
        self._write(f"{cap.speaker} ", f"spk{idx % len(SPEAKER_COLORS)}")
        self._write(f"{cap.source}\n", "src")
        self._write(f"      {cap.translated}\n", "dst")
        if cap.audio:
            self._write(f"      ♪ {cap.audio}\n", "ts")
        self.text.see("end")

    def _write(self, s: str, tag: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", s, tag)
        self.text.configure(state="disabled")

    def _on_close(self) -> None:
        self.stop_flag.set()
        if self.session:
            self.session.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"[{m:02d}:{s:02d}]"


def launch(cfg: Config, window: float = 1.5, device: int | None = None) -> None:
    App(cfg, window=window, device=device).run()
