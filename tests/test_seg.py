from __future__ import annotations

import math
import struct
import unittest

from voicetwin.seg import FRAME_MS, RATE, RingBuffer, Segmenter, overlap_trim

FRAME = RATE * 2 * FRAME_MS // 1000


def tone(seconds: float, amplitude: float = 0.3, freq: float = 220.0) -> bytes:
    count = int(RATE * seconds)
    out = bytearray()
    for i in range(count):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / RATE))
        out.extend(struct.pack("<h", value))
    return bytes(out)


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(RATE * seconds)


class SegmenterTest(unittest.TestCase):
    def test_silence_emits_nothing(self) -> None:
        seg = Segmenter()
        self.assertEqual(seg.feed(silence(1.0)), [])

    def test_speech_then_pause_emits_one_span(self) -> None:
        seg = Segmenter()
        spans = seg.feed(tone(0.8) + silence(0.4))
        self.assertEqual(len(spans), 1)
        self.assertGreater(spans[0].voiced, 0.5)
        self.assertLess(spans[0].seconds, 1.2)

    def test_short_blip_is_dropped(self) -> None:
        seg = Segmenter(min_speech=0.15)
        spans = seg.feed(tone(0.06) + silence(0.6))
        self.assertEqual(spans, [])

    def test_max_segment_forces_cut(self) -> None:
        # 连续说话没有静音：到 hard_max 必须硬切，不能无限等下去
        seg = Segmenter(max_segment=1.0, hard_max=1.0, min_silence=0.5)
        spans = seg.feed(tone(3.0))
        self.assertGreaterEqual(len(spans), 2)
        for span in spans:
            self.assertLessEqual(span.seconds, 1.05)

    def test_prefers_silence_over_hard_cut(self) -> None:
        # 有静音时应该切在静音上，而不是硬切
        seg = Segmenter(max_segment=1.0, hard_max=4.0, min_silence=0.5)
        audio = tone(0.9) + silence(0.15) + tone(0.9) + silence(0.6)
        spans = seg.feed(audio)
        self.assertGreaterEqual(len(spans), 1)
        self.assertLessEqual(spans[0].seconds, 1.2)

    def test_flush_returns_tail(self) -> None:
        seg = Segmenter()
        seg.feed(tone(0.5))
        span = seg.flush()
        self.assertIsNotNone(span)
        self.assertGreater(span.voiced if span else 0, 0.3)

    def test_partial_window_returns_recent_audio(self) -> None:
        seg = Segmenter()
        seg.feed(tone(1.0))
        window = seg.partial_window(0.4)
        self.assertLessEqual(len(window), int(0.4 * RATE * 2) + 2)
        self.assertGreater(len(window), 0)

    def test_arbitrary_chunk_sizes(self) -> None:
        seg = Segmenter()
        audio = tone(0.5) + silence(0.3)
        spans = []
        step = 731  # 故意不对齐 20ms
        for i in range(0, len(audio), step):
            spans.extend(seg.feed(audio[i : i + step]))
        self.assertEqual(len(spans), 1)


class RingBufferTest(unittest.TestCase):
    def test_tail_keeps_latest(self) -> None:
        ring = RingBuffer(seconds=1.0)
        ring.push(silence(2.0))
        self.assertLessEqual(len(ring.tail(0.5)), int(0.5 * RATE * 2) + 2)

    def test_since_tracks_marks(self) -> None:
        ring = RingBuffer(seconds=5.0)
        ring.push(silence(0.2))
        mark = ring.pushed
        ring.push(tone(0.1))
        chunk = ring.since(mark)
        self.assertGreater(len(chunk), 0)
        self.assertLessEqual(len(chunk), int(0.1 * RATE * 2) + 2)


class OverlapTrimTest(unittest.TestCase):
    def test_removes_repeated_english_prefix(self) -> None:
        self.assertEqual(overlap_trim("we translate the person", "the person instead"), "instead")

    def test_removes_repeated_chinese_prefix(self) -> None:
        self.assertEqual(overlap_trim("翻译的不是话", "不是话是说话的人"), "是说话的人")

    def test_keeps_unrelated_text(self) -> None:
        self.assertEqual(overlap_trim("hello there", "完全不同"), "完全不同")


if __name__ == "__main__":
    unittest.main()
