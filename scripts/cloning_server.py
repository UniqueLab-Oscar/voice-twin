"""克隆 TTS 的参考实现：一个极小的 HTTP 服务，实现 VoiceTwin 的合成契约。

契约（VoiceTwin -> 本服务）
    POST /synthesize
    {"text": "要念的中文", "reference_audio": "/abs/ref.wav", "language": "zh"}
    -> 200, Content-Type: audio/wav, body 为 wav 字节

这样 VoiceTwin 不需要知道底下的引擎是 CosyVoice2、GPT-SoVITS 还是别的：
把它们自带的 api server 包一层，或者在 --tts-cmd 里直接调。

用法：
    python scripts/cloning_server.py --engine tone  --port 9880
    python scripts/cloning_server.py --engine sapi  --port 9880   # Windows 自带音色，不换声
    python scripts/cloning_server.py --engine cmd --cmd "python cosy.py --text {text} --ref {ref} --out {out}"

注意：本文件的 tone/sapi 后端只是「把接口跑通」，不产生克隆音色。
真正换声需要 GPU 上的 CosyVoice2 / GPT-SoVITS，见 docs/ARCHITECTURE.md。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SR = 16000


def write_tone(text: str) -> bytes:
    """按字数写一段正弦波：只为验证契约与落盘，不代表音色。"""
    seconds = max(0.4, min(6.0, len(text) * 0.12))
    n = int(SR * seconds)
    amp = 5000
    buf = bytearray()
    for i in range(n):
        v = int(amp * math.sin(2 * math.pi * 220 * i / SR))
        buf += int(v).to_bytes(2, "little", signed=True)
    tmp = Path(tempfile.mkdtemp()) / "tone.wav"
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(buf))
    return tmp.read_bytes()


def synth_sapi(text: str) -> bytes:
    import tempfile as tf

    out = Path(tf.mkdtemp()) / "sapi.wav"
    quote = lambda s: s.replace("'", "''")  # noqa: E731
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SetOutputToWaveFile('{quote(str(out.resolve()))}');"
        f"$s.Speak('{quote(text)}');"
        "$s.Dispose();"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=False)
    return out.read_bytes() if out.exists() else b""


def synth_cmd(template: str, text: str, ref: str) -> bytes:
    out = Path(tempfile.mkdtemp()) / "cmd.wav"
    cmd = template.format(text=text.replace('"', "'"), ref=ref, out=str(out))
    subprocess.run(cmd, shell=True, check=True)
    return out.read_bytes() if out.exists() else b""


class Handler(BaseHTTPRequestHandler):
    engine = "tone"
    cmd = ""

    def log_message(self, *args) -> None:  # noqa: D401
        pass

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/synthesize", "/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload["text"]
            ref = payload.get("reference_audio", "")
        except Exception as exc:  # noqa: BLE001
            self.send_error(400, str(exc))
            return

        if self.engine == "tone":
            data = write_tone(text)
        elif self.engine == "sapi":
            data = synth_sapi(text)
        elif self.engine == "cmd":
            data = synth_cmd(self.cmd, text, ref)
        else:
            self.send_error(500, f"unknown engine: {self.engine}")
            return

        if not data:
            self.send_error(500, "empty audio")
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tone", choices=["tone", "sapi", "cmd"])
    ap.add_argument("--cmd", default="", help="engine=cmd 时的命令模板，支持 {text} {ref} {out}")
    ap.add_argument("--port", type=int, default=9880)
    args = ap.parse_args()
    if args.engine == "cmd" and not args.cmd:
        print("engine=cmd 需要 --cmd", file=sys.stderr)
        return 2

    Handler.engine = args.engine
    Handler.cmd = args.cmd
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"克隆 TTS 参考服务：http://127.0.0.1:{args.port}/synthesize  (engine={args.engine})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
