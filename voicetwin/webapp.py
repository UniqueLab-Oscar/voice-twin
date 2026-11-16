"""实时同传控制台：Python 采音 + 流式识别，浏览器只负责好看地显示。

为什么不做「浏览器采音」：麦克风在 Python 侧拿，链路少一跳（省 30–60ms），
也让同一套逻辑同时喂给 CLI、桌面窗口和这个页面。

接口：
- `GET  /`                  控制台页面
- `GET  /api/devices`       可用输入设备
- `GET  /api/live/events`   SSE：字幕流（partial / final / status）
- `POST /api/live/start`    开始监听
- `POST /api/live/stop`     停止
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import Config
from .live import LiveSession, list_devices, run_mic


def _pick_translator(requested: str | None) -> str:
    choice = (requested or "").strip() or _default_translator()
    if choice == "ollama":
        model = ollama_model("")
        if model:
            _ACTIVE_OLLAMA_MODEL[0] = model
        else:
            return "marian" if _onnx_ready() else "argos"
    return choice


_ACTIVE_OLLAMA_MODEL = [""]


def _onnx_ready() -> bool:
    try:
        from . import mt as mt_mod

        return mt_mod.onnx_dir() is not None
    except Exception:  # noqa: BLE001
        return False


def ollama_model(preferred: str = "") -> str:
    """返回本机 Ollama 可用的模型名；不可用返回空串。"""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    names = [m.get("name", "") for m in tags.get("models", []) if m.get("name")]
    if not names:
        return ""
    if preferred and any(n == preferred or n.startswith(preferred.split(":")[0]) for n in names):
        for n in names:
            if n == preferred:
                return n
    return names[0]


def _default_translator() -> str:
    """本地 Ollama 在线优先（质量最好）；否则本地 opus-mt（ONNX）；再退 argos。"""
    try:
        import json
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        if tags.get("models"):
            return "ollama"
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import mt as mt_mod

        if mt_mod.onnx_dir() is not None:
            return "marian"
        if mt_mod.available():
            return "marian"
    except Exception:  # noqa: BLE001
        pass
    return "argos"


class Hub:
    """全局只跑一路监听；订阅者通过队列拿事件。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.subs: list[queue.Queue[dict]] = []
        self.session: LiveSession | None = None
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.started_at = 0.0
        self.state = "idle"
        self.last: dict | None = None
        self.history: list[dict] = []

    # ---------- 订阅 ----------

    def subscribe(self) -> queue.Queue[dict]:
        q: queue.Queue[dict] = queue.Queue(maxsize=200)
        with self._lock:
            self.subs.append(q)
        for item in list(self.history):
            q.put(item)
        if self.state == "running":
            q.put({"type": "status", "state": "running"})
        return q

    def unsubscribe(self, q: queue.Queue[dict]) -> None:
        with self._lock:
            if q in self.subs:
                self.subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    # ---------- 生命周期 ----------

    @property
    def running(self) -> bool:
        return self.state == "running"

    def start(self, cfg: Config, device: int | None, demo: str = "") -> None:
        with self._lock:
            if self.running:
                return
            self.state = "starting"

        def on_caption(cap) -> None:
            # 同一句话被端点切成几段时，去掉完全重复的（含只有后缀增量的重复）
            src = (cap.source or "").strip()
            if self.history:
                last = self.history[-1]
                if src == last.get("source", "").strip():
                    return
                if last.get("source", "").strip().startswith(src) and len(src) < len(last["source"]):
                    return
            event = {
                "type": "final",
                "t0": round(cap.t0, 2),
                "speaker": cap.speaker,
                "source": cap.source,
                "translated": cap.translated,
                "asr_ms": round(cap.asr_ms),
                "total_ms": round(cap.total_ms),
                "ts": time.time(),
            }
            self.last = event
            with self._lock:
                self.history.append(event)
                self.history = self.history[-200:]
            self.publish(event)

        def on_partial(text: str, lang: str) -> None:
            self.publish({"type": "partial", "text": text, "lang": lang})

        session = LiveSession(cfg, on_caption=on_caption, on_partial=on_partial)
        self.session = session
        self.stop_event = threading.Event()
        self.started_at = time.time()

        def run() -> None:
            self.publish({"type": "status", "state": "running"})
            try:
                assert self.stop_event is not None
                if demo:
                    from .live import replay_wav

                    for pcm in replay_wav(demo, 0.1, realtime=True):
                        if self.stop_event.is_set():
                            break
                        session.feed(pcm)
                    session.flush()
                else:
                    run_mic(session, device=device, stop=self.stop_event.is_set)
            except Exception as exc:  # noqa: BLE001
                self.publish({"type": "error", "message": str(exc)})
            finally:
                try:
                    session.close()
                except Exception:  # noqa: BLE001
                    pass
                self.state = "idle"
                self.publish({"type": "status", "state": "idle"})

        self.state = "running"
        self.thread = threading.Thread(target=run, daemon=True, name="vt-live")
        self.thread.start()

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self.thread = None
        self.session = None
        self.state = "idle"


def _devices() -> list[dict]:
    import sounddevice as sd

    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append({"id": i, "name": dev["name"], "channels": dev["max_input_channels"]})
    return out


INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VoiceTwin · 实时同传</title>
<style>
  :root {
    --bg: #05070d; --panel: rgba(18,22,34,.72); --line: rgba(120,160,255,.14);
    --fg: #e8eefc; --dim: #8b95ad; --accent: #5aa9ff; --ok: #38d39f; --warn: #ffb020;
    --s0: #7cc4ff; --s1: #ffd479; --s2: #b79bff; --s3: #6ee7b7;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: radial-gradient(1200px 700px at 15% -10%, #16224a 0%, transparent 60%),
                radial-gradient(900px 600px at 100% 0%, #1d1236 0%, transparent 55%), var(--bg);
    color: var(--fg); font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    display: flex; flex-direction: column; overflow: hidden;
  }
  header {
    display: flex; align-items: center; gap: 14px; padding: 14px 20px;
    border-bottom: 1px solid var(--line); backdrop-filter: blur(12px);
  }
  .brand { display: flex; align-items: baseline; gap: 10px; }
  .brand b { font-size: 17px; letter-spacing: .3px; }
  .brand span { color: var(--dim); font-size: 12px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #4b5566; }
  .dot.live { background: var(--ok); box-shadow: 0 0 12px var(--ok); animation: pulse 1.6s infinite; }
  @keyframes pulse { 50% { opacity: .45; } }
  .spacer { flex: 1; }
  select, button {
    background: rgba(255,255,255,.05); color: var(--fg); border: 1px solid var(--line);
    border-radius: 10px; padding: 8px 12px; font-size: 13px; outline: none;
  }
  select { max-width: 320px; }
  button { cursor: pointer; transition: .15s; }
  button:hover { border-color: rgba(120,160,255,.5); }
  button.primary { background: linear-gradient(180deg,#2f7fff,#1a5fd6); border: none; font-weight: 600; }
  button.danger { background: linear-gradient(180deg,#ff5f6d,#d63a4a); border: none; font-weight: 600; }
  main { flex: 1; overflow-y: auto; padding: 26px 22px 120px; scroll-behavior: smooth; }
  .wrap { max-width: 1080px; margin: 0 auto; }
  .card {
    border: 1px solid var(--line); background: var(--panel); border-radius: 16px;
    padding: 16px 20px; margin-bottom: 14px; backdrop-filter: blur(10px);
    animation: rise .28s ease-out;
  }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } }
  .meta { display: flex; gap: 10px; align-items: center; color: var(--dim); font-size: 12px; margin-bottom: 8px; }
  .chip { border: 1px solid var(--line); border-radius: 999px; padding: 1px 8px; }
  .src { color: var(--dim); font-size: 14px; line-height: 1.5; }
  .dst { font-size: 21px; line-height: 1.45; margin-top: 6px; }
  .S0 { color: var(--s0); } .S1 { color: var(--s1); } .S2 { color: var(--s2); } .S3 { color: var(--s3); }
  #partial {
    position: fixed; left: 0; right: 0; bottom: 0; padding: 14px 26px 20px;
    background: linear-gradient(180deg, transparent, rgba(5,7,13,.94) 40%);
    color: #aab6d0; font-size: 16px; min-height: 66px; pointer-events: none;
  }
  #partial b { color: #6f7d99; font-weight: 500; font-size: 12px; margin-right: 8px; }
  .empty { color: #5d6880; text-align: center; margin-top: 18vh; font-size: 15px; line-height: 2; }
  .kbd { border: 1px solid var(--line); border-radius: 6px; padding: 1px 6px; font-size: 12px; }
  footer { color: #4d5769; font-size: 12px; text-align: center; padding: 6px; }
</style></head>
<body>
<header>
  <div class="brand"><b>VoiceTwin</b><span>实时字幕同传 · 本地运行</span></div>
  <div class="dot" id="dot"></div>
  <div class="spacer"></div>
  <select id="device"></select>
  <select id="lang">
    <option value="auto">中英自动</option>
    <option value="zh">中文 → English</option>
    <option value="en">English → 中文</option>
  </select>
  <button id="btn" class="primary">▶ 开始</button>
  <button id="demo" title="用示例音频跑一遍，不占用麦克风">试听片段</button>
</header>
<main><div class="wrap" id="list">
  <div class="empty">戴上耳机，点「开始」<br>说话时字幕会一条条浮上来</div>
</div></main>
<div id="partial"></div>
<footer>音频只在本机处理 · <span class="kbd">Esc</span> 停止</footer>
<script>
const list = document.getElementById('list');
const partial = document.getElementById('partial');
const btn = document.getElementById('btn');
const dot = document.getElementById('dot');
const devSel = document.getElementById('device');
const langSel = document.getElementById('lang');
let es = null, running = false;

async function loadDevices() {
  const r = await fetch('/api/devices'); const d = await r.json();
  devSel.innerHTML = d.devices.map(x => `<option value="${x.id}">${x.name}</option>`).join('');
  if (d.best !== undefined) devSel.value = d.best;
}

function card(e) {
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = `<div class="meta">
      <span class="chip">${e.t0.toFixed(1)}s</span>
      <span class="chip">识别 ${e.asr_ms}ms</span>
      <span class="chip">译完 ${e.total_ms}ms</span>
      <span class="${e.speaker}">● ${e.speaker}</span>
    </div>
    <div class="src">${esc(e.source)}</div>
    <div class="dst ${e.speaker}">${esc(e.translated)}</div>`;
  list.appendChild(el);
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function esc(s) { return (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function connect() {
  es = new EventSource('/api/live/events');
  es.onmessage = ev => {
    const e = JSON.parse(ev.data);
    if (e.type === 'final') { partial.innerHTML = ''; card(e); }
    else if (e.type === 'partial') partial.innerHTML = `<b>在说</b>${esc(e.text)}`;
    else if (e.type === 'status') setRunning(e.state === 'running');
    else if (e.type === 'error') partial.innerHTML = `<b>出错</b>${esc(e.message)}`;
  };
  es.onerror = () => { setTimeout(connect, 1200); };
}

function setRunning(on) {
  running = on;
  dot.className = 'dot' + (on ? ' live' : '');
  btn.textContent = on ? '■ 停止' : '▶ 开始';
  btn.className = on ? 'danger' : 'primary';
}

btn.onclick = async () => {
  if (running) { await fetch('/api/live/stop', { method: 'POST' }); setRunning(false); return; }
  const body = { device: Number(devSel.value), lang: langSel.value };
  const r = await fetch('/api/live/start', { method: 'POST', body: JSON.stringify(body) });
  const d = await r.json();
  if (d.ok) { list.innerHTML = ''; setRunning(true); }
  else partial.innerHTML = `<b>启动失败</b>${esc(d.error)}`;
};

document.addEventListener('keydown', e => { if (e.key === 'Escape' && running) btn.click(); });
document.getElementById('demo').onclick = async () => {
  if (running) await fetch('/api/live/stop', { method: 'POST' });
  list.innerHTML = '';
  const r = await fetch('/api/live/start', { method: 'POST', body: JSON.stringify({ demo: 'zh' }) });
  const d = await r.json();
  if (d.ok) setRunning(true); else partial.innerHTML = `<b>失败</b>${esc(d.error)}`;
};
loadDevices(); connect(); setRunning(false);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    hub: Hub

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            data = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/devices":
            devs = _devices()
            best = next(
                (d["id"] for d in devs if any(k in d["name"] for k in ("麦克风", "Microphone", "阵列"))),
                devs[0]["id"] if devs else 0,
            )
            self._json({"devices": devs, "best": best})
            return
        if path == "/api/live/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = self.hub.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        event = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                self.hub.unsubscribe(q)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if path == "/api/live/stop":
            self.hub.stop()
            self._json({"ok": True})
            return
        if path == "/api/live/start":
            lang = payload.get("lang") or "auto"
            demo = payload.get("demo") or ""
            if demo:
                from .paths import demo_dir

                base = demo_dir()
                demo = str(base / {"zh": "sample_zh.wav", "en": "en_speech.wav"}.get(demo, demo))
            cfg = Config(
                mock=False,
                asr_engine="stream",
                seg_mode="vad",
                auto_lang=(lang == "auto"),
                src_lang="zh" if lang == "zh" else "en",
                tgt_lang="en" if lang in ("auto", "zh") else "zh",
                translator=_pick_translator(payload.get("translator")),
                playback=bool(payload.get("playback", True)),
                partial=True,
            )
            if lang in ("zh", "en"):
                cfg.src_lang, cfg.tgt_lang = lang, ("en" if lang == "zh" else "zh")
            if cfg.translator == "ollama" and _ACTIVE_OLLAMA_MODEL[0]:
                cfg.llm_model = _ACTIVE_OLLAMA_MODEL[0]
            try:
                self.hub.start(cfg, device=payload.get("device"), demo=demo)
                self._json({"ok": True})
            except Exception as exc:  # noqa: BLE001
                self._json({"ok": False, "error": str(exc)}, code=500)
            return
        self.send_error(404)


def warmup() -> None:
    """后台预加载翻译模型：第一句话不该等几十秒。"""
    try:
        from . import mt as mt_mod

        where = mt_mod.onnx_dir()
        if where is None:
            return
        from .mt_onnx import translate

        translate(where, "你好", "zh", "en", threads=4)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 翻译模型预热失败：{exc}")


def serve(port: int = 8766, open_browser: bool = True) -> None:
    import threading

    threading.Thread(target=warmup, name="vt-warmup", daemon=True).start()
    hub = Hub()
    handler = type("BoundHandler", (Handler,), {"hub": hub})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"VoiceTwin 控制台：{url}")
    if open_browser:
        import subprocess
        import sys

        edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
        if edge.exists():
            subprocess.Popen([str(edge), f"--app={url}", "--window-size=1180,860"])
        elif sys.platform == "win32":
            import webbrowser

            webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()
        httpd.server_close()
