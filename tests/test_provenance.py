import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetwin.provenance import (  # noqa: E402
    _split_chunks,
    assert_wav_playable,
    build_meta,
    fingerprint,
    read_stamp,
    stamp,
    verify_license_notice,
)
from voicetwin.stages import write_tone  # noqa: E402


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.wav = self.dir / "001.wav"
        write_tone(str(self.wav), 0.3, 440.0)
        self.ref = self.dir / "ref.wav"
        write_tone(str(self.ref), 1.0, 300.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip(self):
        meta = build_meta("cosyvoice2", reference=self.ref, consent="granted")
        stamp(self.wav, meta)
        got = read_stamp(self.wav)
        self.assertIsNotNone(got)
        self.assertEqual(got["engine"], "cosyvoice2")
        self.assertEqual(got["consent"], "granted")
        self.assertEqual(got["reference_sha256"], fingerprint(self.ref))
        self.assertEqual(got["format"], "VTWN1")

    def test_audio_data_untouched(self):
        before = dict(_split_chunks(self.wav.read_bytes())[1])[b"data"]
        stamp(self.wav, build_meta("tone"))
        after_map = dict(_split_chunks(self.wav.read_bytes())[1])
        self.assertEqual(before, after_map[b"data"])

    def test_still_playable(self):
        stamp(self.wav, build_meta("tone"))
        assert_wav_playable(self.wav)
        with wave.open(str(self.wav), "rb") as w:
            self.assertGreater(w.getnframes(), 0)

    def test_unstamped_returns_none(self):
        self.assertIsNone(read_stamp(self.wav))

    def test_restamp_replaces_not_duplicates(self):
        stamp(self.wav, build_meta("tone"))
        stamp(self.wav, build_meta("sapi"))
        chunks = [cid for cid, _ in _split_chunks(self.wav.read_bytes())[1]]
        self.assertEqual(chunks.count(b"vtwn"), 1)
        self.assertEqual(read_stamp(self.wav)["engine"], "sapi")

    def test_consent_validated(self):
        with self.assertRaises(ValueError):
            build_meta("tone", consent="whatever")

    def test_verify_license_notice(self):
        sha = fingerprint(self.ref)
        stamp(self.wav, build_meta("tone", reference=self.ref))
        self.assertTrue(verify_license_notice(self.wav, sha))
        self.assertFalse(verify_license_notice(self.wav, "0" * 64))

    def test_rejects_non_riff(self):
        bad = self.dir / "bad.wav"
        bad.write_bytes(b"not a wav file at all")
        self.assertIsNone(read_stamp(bad))
        with self.assertRaises(ValueError):
            stamp(bad, build_meta("tone"))


if __name__ == "__main__":
    unittest.main()
