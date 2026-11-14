from __future__ import annotations

import unittest

from voicetwin.config import Config
from voicetwin.translate import TranslateCache, Translator, numbers_of


class CacheTest(unittest.TestCase):
    def test_put_and_get(self) -> None:
        cache = TranslateCache(maxsize=2)
        key = cache.key("hello", "en", "zh")
        self.assertIsNone(cache.get(key))
        cache.put(key, "你好")
        self.assertEqual(cache.get(key), "你好")
        self.assertEqual(cache.hits, 1)

    def test_eviction_keeps_recent(self) -> None:
        cache = TranslateCache(maxsize=2)
        keys = [cache.key(f"s{i}", "en", "zh") for i in range(3)]
        for i, key in enumerate(keys):
            cache.put(key, str(i))
        self.assertIsNone(cache.get(keys[0]))
        self.assertEqual(cache.get(keys[2]), "2")

    def test_key_differs_by_direction(self) -> None:
        cache = TranslateCache()
        self.assertNotEqual(cache.key("hi", "en", "zh"), cache.key("hi", "zh", "en"))


class NumbersTest(unittest.TestCase):
    def test_extracts_numbers_and_units(self) -> None:
        self.assertEqual(numbers_of("会议 3:30 开始，预算 12.5%"), ["3:30", "12.5%"])

    def test_empty(self) -> None:
        self.assertEqual(numbers_of("没有数字"), [])


class ProtectTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        text = "今天 3 点开会，约 45 分钟"
        protected, mapping = Translator._protect(text)
        self.assertNotIn("3", protected)
        restored = Translator._restore(protected, mapping)
        self.assertEqual(restored, text)


class TranslatorTest(unittest.TestCase):
    def test_passthrough_and_cache(self) -> None:
        cfg = Config(translator="passthrough")
        translator = Translator(cfg)
        self.assertEqual(translator.translate("hello"), "hello")
        self.assertEqual(translator.translate("hello"), "hello")
        self.assertEqual(translator.cache.hits, 1)

    def test_context_grows_then_trims(self) -> None:
        cfg = Config(translator="passthrough")
        translator = Translator(cfg, history=2)
        for i in range(5):
            translator.translate(f"line {i}")
        self.assertLessEqual(len(translator.turns), 6)
        self.assertIn("line 3", translator._context())  # noqa: SLF001

    def test_blank_returns_empty(self) -> None:
        self.assertEqual(Translator(Config(translator="passthrough")).translate("   "), "")


if __name__ == "__main__":
    unittest.main()
