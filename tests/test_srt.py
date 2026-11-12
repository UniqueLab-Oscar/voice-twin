import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetwin.srt import to_srt  # noqa: E402
from voicetwin.types import RunReport, Segment, Synthesis, Translated  # noqa: E402


class SrtTest(unittest.TestCase):
    def test_format(self):
        seg = Segment(0.0, 3.2, "hello", "S0")
        report = RunReport(syntheses=[Synthesis(Translated(seg, "你好"), "001.wav", "S0")])
        out = to_srt(report)
        self.assertIn("1\n00:00:00,000 --> 00:00:03,200\n你好", out)

    def test_rounding(self):
        seg = Segment(65.5, 66.25, "x", "S0")
        report = RunReport(syntheses=[Synthesis(Translated(seg, "y"), "", "S0")])
        self.assertIn("00:01:05,500 --> 00:01:06,250", to_srt(report))


if __name__ == "__main__":
    unittest.main()
