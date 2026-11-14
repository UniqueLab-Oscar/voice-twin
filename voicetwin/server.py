"""一个零依赖的本地 Web 演示台。

只依赖标准库：`ThreadingHTTPServer` + 内嵌 HTML/JS。
启动后浏览器里拖一段音频进去，就能看到每个阶段的耗时、译文和可播放的配音。

设计取舍：
- 只监听 127.0.0.1，绝不对局域网暴露（伦理红线：音频不出机器）。
- 上传走 `application/octet-stream` 原始字节，避免手写 multipart 解析。
- 文件名一律取 basename 并做白名单校验，防目录穿越。
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import Config
from .pipeline import Pipeline
from .report import render_report
from .srt import to_srt
from .types import RunReport

MAX_UPLOAD = 200 * 1024 * 1024  # 200 MB，够一段小时级会议录音
SAFE_NAME = re.compile(r"^[A-Za-z0-9._\-\u4e00-\u9fff]+$")


def _safe(name: str) -> str:
    name = Path(name).name
    if name in ("", ".", ".."):
        raise ValueError(f"非法文件名：{name!r}")
    if not SAFE_NAME.match(name):
        raise ValueError(f"非法文件名：{name!r}")
    return name


def render_rows(report: RunReport) -> list[dict]:
    rows: list[dict] = []
    for i, syn in enumerate(report.syntheses, start=1):
        seg = syn.translated.segment
        rows.append(
            {
                "i": i,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "speaker": seg.speaker,
                "source": seg.text,
                "translated": syn.translated.text,
                "audio": Path(syn.audio_path).name if syn.audio_path else "",
                "ok": syn.ok,
            }
        )
    return rows


INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VoiceTwin · 声替</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--acc:#58a6ff;--ok:#3fb950;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{padding:22px 28px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:19px;letter-spacing:.5px}
h1 span{color:var(--acc)}
.sub{color:var(--dim);font-size:12.5px;margin-top:4px}
main{padding:22px 28px;max-width:1080px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
#drop{border:1.5px dashed var(--line);border-radius:8px;padding:18px;text-align:center;color:var(--dim);cursor:pointer;flex:1;min-width:240px}
#drop.on{border-color:var(--acc);color:var(--fg)}
button{background:#21262d;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 14px;cursor:pointer;font-size:13px}
button.primary{background:var(--acc);border-color:var(--acc);color:#04121f;font-weight:600}
button:disabled{opacity:.5;cursor:default}
select,input{background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:13px}
label{color:var(--dim);font-size:12.5px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:500;font-size:12px}
td.src{color:var(--dim)}
.tl{display:flex;gap:8px;flex-wrap:wrap}
.chip{background:#0d1117;border:1px solid var(--line);border-radius:999px;padding:4px 11px;font-size:12px}
.chip b{color:var(--acc);font-weight:600}
.err{color:#f85149;white-space:pre-wrap;font-size:12.5px}
a{color:var(--acc)}
audio{height:30px;max-width:230px}
.hint{color:var(--dim);font-size:12px}
</style></head><body>
<header>
  <h1>VoiceTwin <span>· 声替</span></h1>
  <div class="sub">不是把话翻译成中文，而是让那个人用他自己的声音继续说中文 · 全部本地运行</div>
</header>
<main>
  <div class="panel">
    <div class="row">
      <div id="drop">把音频拖到这里，或点这里选择文件<div class="hint" id="fname">支持 wav / mp3 / m4a（<=200MB）</div></div>
    </div>
    <div class="row" style="margin-top:12px">
      <label>ASR 模型
        <select id="asr"><option>tiny.en</option><option>base</option><option selected>small</option></select>
      </label>
      <label>译员后端
        <select id="translator">
          <option value="ollama">Ollama（本地 LLM）</option>
          <option value="openai">OpenAI 兼容 API</option>
          <option value="argos">Argos（离线 NMT）</option>
          <option value="passthrough">直通（只看识别）</option>
        </select>
      </label>
      <label>风格
        <select id="style"><option value="interpreter" selected>译员式</option><option value="literal">直译</option></select>
      </label>
      <label>TTS
        <select id="tts"><option value="sapi">SAPI（本机自带）</option><option value="cosyvoice2">CosyVoice2（克隆）</option><option value="none">不合成</option></select>
      </label>
      <label>克隆服务地址
        <input id="ttsurl" placeholder="http://127.0.0.1:9880/synthesize" style="width:250px">
      </label>
      <button class="primary" id="go" disabled>开始翻译</button>
      <span class="hint" id="state"></span>
    </div>
  </div>

  <div class="panel" id="timing" style="display:none">
    <div class="tl" id="chips"></div>
  </div>

  <div class="panel" id="result" style="display:none">
    <table><thead><tr><th style="width:52px">#</th><th style="width:70px">时间</th><th style="width:70px">说话人</th>
      <th>原文</th><th>译文</th><th style="width:230px">配音</th></tr></thead><tbody id="rows"></tbody></table>
    <div class="row" style="margin-top:12px">
      <a id="srt" href="#" download>下载 SRT 字幕</a>
      <span class="hint" id="summary"></span>
    </div>
  </div>
  <div class="panel err" id="err" style="display:none"></div>
</main>
<script>
const $ = s => document.querySelector(s);
let picked = null;
const drop = $('#drop');
drop.onclick = () => {
  const i = document.createElement('input');
  i.type = 'file'; i.accept = 'audio/*,.wav,.mp3,.m4a,.flac';
  i.onchange = () => i.files[0] && upload(i.files[0]);
  i.click();
};
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.add('on');}));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.remove('on');}));
drop.addEventListener('drop', ev => { const f = ev.dataTransfer.files[0]; if (f) upload(f); });

async function upload(file) {
  $('#state').textContent = '上传中…';
  const r = await fetch('/api/upload?name=' + encodeURIComponent(file.name), {method:'POST', body:file});
  const j = await r.json();
  if (!j.ok) { fail(j.error); return; }
  picked = j.path;
  $('#fname').textContent = file.name + ' · ' + (file.size/1024/1024).toFixed(2) + ' MB';
  $('#go').disabled = false;
  $('#state').textContent = '';
}

function fail(msg) { $('#err').style.display='block'; $('#err').textContent = msg; }

$('#go').onclick = async () => {
  $('#go').disabled = true; $('#err').style.display='none';
  $('#state').textContent = '跑链路中…（真实 ASR 在 CPU 上可能需要几秒到几十秒）';
  const body = {audio: picked, asr_model: $('#asr').value, translator: $('#translator').value,
                style: $('#style').value, tts: $('#tts').value, tts_url: $('#ttsurl').value.trim()};
  try {
    const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    if (!j.ok) { fail(j.error); return; }
    render(j);
  } catch (e) { fail(String(e)); }
  finally { $('#go').disabled = false; $('#state').textContent = ''; }
};

function render(j) {
  $('#timing').style.display = 'block';
  $('#chips').innerHTML = Object.entries(j.timings).map(([k,v]) =>
    `<span class="chip">${k} <b>${v < 0 ? (k.indexOf('skip') >= 0 ? '未启用' : '失败') : v.toFixed(0) + ' ms'}</b></span>`).join('');
  $('#result').style.display = 'block';
  $('#rows').innerHTML = j.rows.map(r => `<tr>
    <td>${r.i}</td><td>${r.start.toFixed(2)}s</td><td>${r.speaker}</td>
    <td class="src">${esc(r.source)}</td><td>${esc(r.translated)}</td>
    <td>${r.audio ? `<audio controls preload="none" src="/api/audio?name=${encodeURIComponent(r.audio)}&dir=${encodeURIComponent(j.out)}"></audio>` : '<span class="hint">无</span>'}</td>
  </tr>`).join('');
  $('#srt').href = `/api/file?name=report.srt&dir=${encodeURIComponent(j.out)}`;
  $('#summary').textContent = `${j.rows.length} 句 · 总耗时 ${(j.timings.total||0).toFixed(0)} ms · 产物目录 ${j.out}`;
}

function esc(s) { return (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceTwin/0.1"
    out_dir: Path = Path("out/web")
    quiet = True

    # ---- helpers -------------------------------------------------------
    def _json(self, obj: dict, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"ok": False, "error": f"文件不存在：{path.name}"}, 404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        if not self.quiet:
            super().log_message(fmt, *args)

    # ---- routes --------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._html(INDEX_HTML)
            return
        if url.path in ("/api/audio", "/api/file"):
            q = parse_qs(url.query)
            try:
                name = _safe(q.get("name", [""])[0])
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
                return
            sub = q.get("dir", [str(self.out_dir)])[0]
            base = Path(sub).resolve()
            # 只允许读输出目录之下的文件
            root = self.out_dir.resolve()
            if root not in base.parents and base != root:
                self._json({"ok": False, "error": "目录越界"}, 403)
                return
            self._file(base / name)
            return
        self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            self._json({"ok": False, "error": "文件超过 200MB"}, 413)
            return
        body = self.rfile.read(length)

        if url.path == "/api/upload":
            q = parse_qs(url.query)
            try:
                name = _safe(q.get("name", ["audio.wav"])[0])
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
                return
            dest_dir = self.out_dir / "uploads"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / name
            dest.write_bytes(body)
            self._json({"ok": True, "path": str(dest)})
            return

        if url.path == "/api/run":
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json({"ok": False, "error": "请求体不是合法 JSON"}, 400)
                return
            try:
                result = self._run(payload)
            except Exception as exc:  # noqa: BLE001 - 返回给前端展示
                self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            self._json(result)
            return

        self._json({"ok": False, "error": "not found"}, 404)

    # ---- 真正跑链路 -----------------------------------------------------
    def _run(self, payload: dict) -> dict:
        audio = (payload.get("audio") or "").strip()
        if not audio or not Path(audio).is_file():
            raise FileNotFoundError(f"音频不存在：{audio!r}")
        tts = payload.get("tts", "cosyvoice2")
        cfg = Config(
            out_dir=str(self.out_dir),
            mock=False,
            asr_model=payload.get("asr_model", "small"),
            translator=payload.get("translator", "ollama"),
            style=payload.get("style", "interpreter"),
            tts_engine=tts if tts != "none" else "none",
            tts_url=(payload.get("tts_url") or "").strip(),
        )
        if tts == "none":
            cfg.tts_cmd = ""
        report = Pipeline(cfg).run(audio, audio_path=audio)

        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(render_report(report), encoding="utf-8")
        (out / "report.srt").write_text(to_srt(report), encoding="utf-8")
        return {
            "ok": True,
            "rows": render_rows(report),
            "timings": report.elapsed_ms,
            "out": str(out),
        }


def serve(port: int = 8765, out_dir: str = "out/web", open_browser: bool = True) -> None:
    Handler.out_dir = Path(out_dir)
    Handler.out_dir.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"VoiceTwin 演示台：{url}")
    print("按 Ctrl+C 停止。音频只在本机流转。")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
