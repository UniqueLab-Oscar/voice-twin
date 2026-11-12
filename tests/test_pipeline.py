import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetwin.config import Config  # noqa: E402
from voicetwin.pipeline import Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO = str(ROOT / "demo" / "transcript.tsv")


class MockPipelineTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(mock=True, out_dir=tmp)
            report = Pipeline(cfg).run(DEMO)

            self.assertEqual(len(report.segments), 4)
            self.assertEqual(len(report.syntheses), 4)
            self.assertEqual(report.syntheses[0].translated.text, "实时翻译一直只在处理『词』。")
            self.assertEqual(report.syntheses[0].translated.segment.speaker, "S0")
            self.assertEqual(report.syntheses[2].translated.segment.speaker, "S1")
            for syn in report.syntheses:
                self.assertTrue(Path(syn.audio_path).exists())
                self.assertGreater(Path(syn.audio_path).stat().st_size, 1000)
            for key in ("asr", "translate", "tts", "total"):
                self.assertIn(key, report.elapsed_ms)

    def test_literal_and_interpreter_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            lit = Pipeline(Config(mock=True, out_dir=tmp, style="literal")).run(DEMO)
            itp = Pipeline(Config(mock=True, out_dir=tmp, style="interpreter")).run(DEMO)

            self.assertEqual(len(lit.translations), len(itp.translations))
            self.assertNotEqual(lit.translations[0].text, itp.translations[0].text)
            # 译员版更短：压缩了填充词
            self.assertLess(len(itp.translations[0].text), len(lit.translations[0].text))

    def test_glossary_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = Path(tmp) / "glossary.txt"
            g.write_text("实时翻译=VoiceTwin\n", encoding="utf-8")
            report = Pipeline(Config(mock=True, out_dir=tmp, glossary=str(g))).run(DEMO)
            self.assertTrue(report.translations[0].text.startswith("VoiceTwin"))

    def test_subtitle_only_keeps_rows(self):
        """关掉配音（tts_engine='none'）时仍要保留每句译文，否则前端会显示成空。"""
        with tempfile.TemporaryDirectory() as tmp:
            report = Pipeline(Config(mock=True, out_dir=tmp, tts_engine="none")).run(DEMO)
            self.assertEqual(len(report.syntheses), len(report.translations))
            self.assertEqual(len(report.syntheses), 4)
            for syn in report.syntheses:
                self.assertFalse(syn.ok)
                self.assertEqual(syn.audio_path, "")
                self.assertTrue(syn.translated.text)


if __name__ == "__main__":
    unittest.main()
