"""克隆 TTS 适配层与 Web 服务工具函数的测试（不依赖任何模型）。"""

from __future__ import annotations

import json
import math
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from voicetwin.config import Config
from voicetwin.server import _safe, render_rows
from voicetwin.stages import VoiceTts, extract_reference
from voicetwin.types import RunReport, Segment, Synthesis, Translated


def _write_wav(path: Path, seconds: float = 12.0, sr: int = 16000) -> None:
    n = int(seconds * sr)
    buf = bytearray()
    for i in range(n):
        buf += int(3000 * math.sin(2 * math.pi * 200 * i / sr)).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(buf))


class _StubTts(BaseHTTPRequestHandler):
    payload_seen: dict = {}

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        _StubTts.payload_seen = json.loads(self.rfile.read(length).decode("utf-8"))
        data = b"RIFF" + b"\x00" * 100  # 内容不重要，验证「字节原样落盘」
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestReference(unittest.TestCase):
    def test_extract_reference_trims_to_n_seconds(self) -> None:
        with TemporaryDirectory() as td:
            src = Path(td) / "src.wav"
            _write_wav(src, 20.0)
            ref = extract_reference(str(src), Path(td) / "ref.wav", seconds=5.0)
            with wave.open(ref, "rb") as w:
                self.assertAlmostEqual(w.getnframes() / w.getframerate(), 5.0, places=2)

    def test_extract_reference_rejects_non_wav(self) -> None:
        with TemporaryDirectory() as td:
            src = Path(td) / "src.mp3"
            src.write_bytes(b"not really mp3")
            with self.assertRaises(RuntimeError):
                extract_reference(str(src), Path(td) / "ref.wav")


class TestHttpTts(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubTts)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_synthesize_posts_contract_and_writes_bytes(self) -> None:
        with TemporaryDirectory() as td:
            cfg = Config(
                out_dir=td,
                mock=False,
                tts_engine="http",
                tts_url=f"http://127.0.0.1:{self.port}/synthesize",
                ref_audio="C:/ref.wav",
                tgt_lang="zh",
            )
            tr = Translated(Segment(0.0, 2.0, "你好，世界。", "S1"), "你好，世界。")
            syn = VoiceTts(cfg).synthesize(tr, 1)
            self.assertTrue(syn.ok)
            self.assertTrue(Path(syn.audio_path).exists())
            self.assertEqual(_StubTts.payload_seen["text"], "你好，世界。")
            self.assertEqual(_StubTts.payload_seen["reference_audio"], "C:/ref.wav")
            self.assertEqual(_StubTts.payload_seen["language"], "zh")

    def test_missing_engine_raises_readable_error(self) -> None:
        with TemporaryDirectory() as td:
            cfg = Config(out_dir=td, mock=False, tts_engine="cosyvoice2")
            tr = Translated(Segment(0.0, 1.0, "hi", "S0"), "嗨")
            with self.assertRaises(RuntimeError) as ctx:
                VoiceTts(cfg).synthesize(tr, 1)
            self.assertIn("--tts-url", str(ctx.exception))


class TestServerHelpers(unittest.TestCase):
    def test_safe_normalizes_paths_to_basename(self) -> None:
        # 目录穿越被「取 basename」消解，只会落在输出目录里
        self.assertEqual(_safe("../secret.wav"), "secret.wav")
        self.assertEqual(_safe("a/b.wav"), "b.wav")

    def test_safe_rejects_dot_and_symbols(self) -> None:
        for bad in ("..", ".", "x;rm.wav", "a b.wav"):
            with self.assertRaises(ValueError):
                _safe(bad)

    def test_safe_keeps_plain_names(self) -> None:
        self.assertEqual(_safe("report.srt"), "report.srt")

    def test_render_rows_uses_basename(self) -> None:
        seg = Segment(0.0, 1.5, "你好", "S0")
        syn = Synthesis(Translated(seg, "你好"), "C:/out/001.wav", "S0")
        report = RunReport(segments=[seg], translations=[syn.translated], syntheses=[syn])
        rows = render_rows(report)
        self.assertEqual(rows[0]["audio"], "001.wav")
        self.assertEqual(rows[0]["speaker"], "S0")
        self.assertEqual(rows[0]["translated"], "你好")


if __name__ == "__main__":
    unittest.main()
