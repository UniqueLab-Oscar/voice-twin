import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetwin.timing import (  # noqa: E402
    duration_seconds,
    fit_to_timeline,
    trim_silence,
)


def _write(path: Path, samples: array, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def _tone(seconds: float, rate: int = 16000, amplitude: int = 8000) -> array:
    import math

    n = int(seconds * rate)
    return array("h", [int(amplitude * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)])


class TimingTest(unittest.TestCase):
    def test_trim_silence_strips_edges(self):
        rate = 16000
        lead = array("h", bytes(2 * int(0.3 * rate)))
        body = _tone(0.5, rate)
        tail = array("h", bytes(2 * int(0.4 * rate)))
        samples = lead + body + tail

        trimmed = trim_silence(samples, rate)
        self.assertGreater(len(trimmed), 0)
        self.assertLess(len(trimmed), len(samples) - int(0.5 * rate))
        self.assertAlmostEqual(len(trimmed) / rate, 0.5, delta=0.05)

    def test_pads_short_audio_to_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            _write(src, _tone(0.4))
            fit_to_timeline(src, dst, 2.0)
            self.assertAlmostEqual(duration_seconds(dst), 2.0, delta=0.02)

    def test_truncates_long_audio_to_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            _write(src, _tone(3.0))
            fit_to_timeline(src, dst, 1.0, max_speed=1.3)
            self.assertAlmostEqual(duration_seconds(dst), 1.0, delta=0.02)

    def test_speed_cap_keeps_audio_when_target_tiny(self):
        """目标时长太短时只加速到上限，剩余部分淡出截断，不无限变速。"""
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            _write(src, _tone(4.0))
            fit_to_timeline(src, dst, 0.5, max_speed=1.3)
            self.assertAlmostEqual(duration_seconds(dst), 0.5, delta=0.02)

    def test_output_is_mono_16bit_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            _write(src, _tone(1.0))
            fit_to_timeline(src, dst, 1.0)
            with wave.open(str(dst), "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 16000)


if __name__ == "__main__":
    unittest.main()
