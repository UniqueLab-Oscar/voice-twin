"""分块同传的切分与延迟统计测试（不加载任何模型）。"""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from voicetwin.stream import StreamRunner, render_srt, render_stream, split_wav
from voicetwin.types import Segment, Translated


def _write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)


class SplitWavTest(unittest.TestCase):
    def test_splits_evenly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.wav"
            _write_wav(src, 6.0)
            chunks = split_wav(src, Path(tmp) / "out", window=2.0)
            self.assertEqual(len(chunks), 3)
            self.assertAlmostEqual(chunks[0].start, 0.0)
            self.assertAlmostEqual(chunks[1].start, 2.0)
            self.assertAlmostEqual(chunks[2].end, 6.0)
            for c in chunks:
                self.assertTrue(c.path.exists())
                self.assertAlmostEqual(c.seconds, 2.0, places=2)

    def test_short_tail_merges_into_last_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "b.wav"
            _write_wav(src, 3.2)
            chunks = split_wav(src, Path(tmp) / "out", window=1.0)
            # 1.0 + 1.0 + 1.2：0.2s 的尾巴不单独成块
            self.assertEqual(len(chunks), 3)
            self.assertAlmostEqual(chunks[-1].seconds, 1.2, places=2)
            self.assertAlmostEqual(chunks[-1].end, 3.2, places=2)

    def test_invalid_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "c.wav"
            _write_wav(src, 1.0)
            with self.assertRaises(ValueError):
                split_wav(src, Path(tmp) / "out", window=0)


class StreamRunnerTest(unittest.TestCase):
    def test_offsets_segments_to_absolute_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "d.wav"
            _write_wav(src, 4.0)
            chunks = split_wav(src, Path(tmp) / "out", window=2.0)
            seen: list[float] = []

            def fake_asr(path: str, prompt: str = "") -> list[Segment]:
                return [Segment(0.0, 1.0, f"chunk-{Path(path).stem}")]

            def fake_tr(segs: list[Segment]) -> list[Translated]:
                return [Translated(s, s.text.upper()) for s in segs]

            report = StreamRunner(fake_asr, fake_tr, on_event=lambda e: seen.append(e.chunk.start)).run(chunks)
            self.assertEqual(len(report.events), 2)
            # 块内 0.0 秒要挪回整段音频的 0.0 / 2.0 秒
            self.assertEqual([e.segments[0].start for e in report.events], [0.0, 2.0])
            self.assertEqual(seen, [0.0, 2.0])
            self.assertAlmostEqual(report.audio_seconds, 4.0, places=2)
            self.assertEqual(len(report.translations), 2)

    def test_empty_chunk_has_no_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "e.wav"
            _write_wav(src, 2.0)
            chunks = split_wav(src, Path(tmp) / "out", window=2.0)
            report = StreamRunner(lambda _p, _q="": [], lambda segs: []).run(chunks)
            self.assertEqual(len(report.events), 1)
            self.assertEqual(report.events[0].translations, [])

    def test_renderers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "f.wav"
            _write_wav(src, 2.0)
            chunks = split_wav(src, Path(tmp) / "out", window=2.0)
            report = StreamRunner(
                lambda _p, _q="": [Segment(0.0, 1.0, "hello")],
                lambda segs: [Translated(s, "你好") for s in segs],
            ).run(chunks)
            srt = render_srt(report)
            self.assertIn("你好", srt)
            self.assertIn("-->", srt)
            md = render_stream(report, 2.0)
            self.assertIn("RTF", md)
            self.assertIn("你好", md)


if __name__ == "__main__":
    unittest.main()
