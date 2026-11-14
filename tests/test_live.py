"""实时字幕与会话导出的纯逻辑测试（不碰麦克风与模型）。"""

import unittest

from voicetwin.live import Caption, captions_to_srt


class TestCaptions(unittest.TestCase):
    def test_srt_has_index_timing_and_speaker(self) -> None:
        caps = [
            Caption(1, 0.0, 2.5, "S0", "hello", "你好"),
            Caption(2, 2.5, 4.0, "S1", "world", "世界"),
        ]
        srt = captions_to_srt(caps)
        self.assertIn("00:00:00,000 --> 00:00:02,500", srt)
        self.assertIn("[S0] 你好", srt)
        self.assertIn("[S1] 世界", srt)
        self.assertTrue(srt.startswith("1\n"))

    def test_empty_session_gives_empty_srt(self) -> None:
        self.assertEqual(captions_to_srt([]), "")


if __name__ == "__main__":
    unittest.main()
