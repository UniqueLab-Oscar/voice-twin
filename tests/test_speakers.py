"""说话人区分的纯算法测试：不依赖麦克风和模型。"""

import math
import struct
import unittest
from array import array

from voicetwin.speakers import SpeakerTracker, median_f0, pcm_to_floats, rms


def tone(freq: float, seconds: float = 0.8, rate: int = 16000, amp: float = 0.4) -> bytes:
    n = int(seconds * rate)
    frames = array("h", (int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)))
    return frames.tobytes()


def silence(seconds: float = 0.8, rate: int = 16000) -> bytes:
    return b"\x00\x00" * int(seconds * rate)


class TestFeatures(unittest.TestCase):
    def test_pcm_to_floats_range(self) -> None:
        values = pcm_to_floats(struct.pack("<hhh", 0, 16384, -32768))
        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(values[1], 0.5, places=3)
        self.assertAlmostEqual(values[2], -1.0, places=3)

    def test_rms_of_silence_is_zero(self) -> None:
        self.assertEqual(rms(pcm_to_floats(silence())), 0.0)

    def test_median_f0_finds_tone(self) -> None:
        found = median_f0(pcm_to_floats(tone(200)), 16000)
        self.assertAlmostEqual(found, 200, delta=8)

    def test_median_f0_ignores_silence(self) -> None:
        self.assertEqual(median_f0(pcm_to_floats(silence()), 16000), 0.0)


class TestTracker(unittest.TestCase):
    def test_same_voice_keeps_one_speaker(self) -> None:
        tracker = SpeakerTracker()
        names = {tracker.assign(tone(180)) for _ in range(3)}
        self.assertEqual(len(tracker.speakers), 1)
        self.assertEqual(names, {"S0"})

    def test_new_pitch_after_confirm_creates_speaker(self) -> None:
        tracker = SpeakerTracker(confirm_hits=2)
        tracker.assign(tone(150))
        first = tracker.assign(tone(150))
        second = tracker.assign(tone(260))
        third = tracker.assign(tone(260))
        self.assertEqual(first, "S0")
        self.assertNotEqual(third, first)
        self.assertEqual(len(tracker.speakers), 2)
        self.assertEqual(second, "S0")  # 第一次像新声音先不建档

    def test_silence_keeps_last_speaker(self) -> None:
        tracker = SpeakerTracker()
        tracker.assign(tone(150))
        name = tracker.assign(silence())
        self.assertEqual(name, "S0")

    def test_reference_returns_first_chunk(self) -> None:
        tracker = SpeakerTracker()
        pcm = tone(170)
        tracker.assign(pcm)
        tracker.assign(pcm)
        ref, rate = tracker.reference("S0")
        self.assertGreater(len(ref), 0)
        self.assertEqual(rate, 16000)

    def test_unknown_reference_is_empty(self) -> None:
        tracker = SpeakerTracker()
        ref, rate = tracker.reference("S9")
        self.assertEqual(ref, b"")
        self.assertEqual(rate, 16000)

    def test_summary_mentions_hz(self) -> None:
        tracker = SpeakerTracker()
        tracker.assign(tone(190))
        tracker.assign(tone(190))
        self.assertIn("Hz", tracker.summary())


if __name__ == "__main__":
    unittest.main()
