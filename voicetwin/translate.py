"""译员层：上下文、缓存、术语、数字校验。

为什么单独一层：``stages.Interpreter`` 是「一句一句翻」的通用实现，实时场景需要
额外三件事——

1. **上下文**：把前一两句（原文+译文）一起送进去，指代和话题才不会断；
2. **缓存**：同一句重复出现（会展、回放）直接命中，0ms；
3. **数字校验**：会议里最不能错的就是数字和单位，源句里的数字在译文中必须都在。

后端沿用 ``Interpreter``：ollama / openai / argos / passthrough。
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass

from .config import Config
from .stages import Interpreter

NUMBER_RE = re.compile(r"\d[\d.,:/%-]*")

PROMPT = """你是资深会议同传译员。把下面这句 {src} 译成 {tgt}。
要求：
- 口语、短句，可以直接念出来；
- 数字、单位、人名、机构名、产品名原样保留；
- 不解释、不扩写、不加引号、不写「译：」；
- 只输出译文本身。
{glossary}{context}
待译：{text}
"""


@dataclass
class Turn:
    """一句已完成的对译，用作上下文。"""

    source: str
    translated: str


class TranslateCache:
    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, str] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, text: str, src: str, tgt: str) -> str:
        raw = f"{src}|{tgt}|{text.strip()}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        value = self._data.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def numbers_of(text: str) -> list[str]:
    return NUMBER_RE.findall(text or "")


class Translator:
    """带上下文与缓存的译员。"""

    def __init__(self, cfg: Config, history: int | None = None) -> None:
        self.cfg = cfg
        self.backend = Interpreter(cfg)
        if history is None:
            history = 2 if getattr(cfg, "quality", "fast") == "best" else 1
        self.history = history
        self.cache = TranslateCache()
        self.turns: list[Turn] = []

    # ---------- 对外 ----------

    def translate(self, text: str, src: str | None = None, tgt: str | None = None) -> str:
        src = src or self.cfg.src_lang
        tgt = tgt or self.cfg.tgt_lang
        text = (text or "").strip()
        if not text:
            return ""

        key = self.cache.key(text, src, tgt)
        cached = self.cache.get(key)
        if cached is not None:
            self._remember(text, cached)
            return cached

        if self.cfg.translator == "passthrough":
            out = text
        elif self.cfg.translator == "argos":
            out = self._argos(text)
        else:
            out = self._llm(text, src, tgt)

        out = (out or "").strip()
        missing = [n for n in numbers_of(text) if n not in out]
        if missing and self.cfg.translator in {"ollama", "openai"}:
            out = self._llm(text, src, tgt, strict_numbers=True) or out

        if out:
            self.cache.put(key, out)
            self._remember(text, out)
        return out

    # ---------- 内部 ----------

    def _remember(self, text: str, out: str) -> None:
        self.turns.append(Turn(text, out))
        if len(self.turns) > 6:
            del self.turns[0]

    def _context(self) -> str:
        if not self.turns or self.history <= 0:
            return ""
        lines = ["\n前文参考（保持术语与指代一致）："]
        for turn in self.turns[-self.history :]:
            lines.append(f"  原：{turn.source}")
            lines.append(f"  译：{turn.translated}")
        return "\n".join(lines) + "\n"

    def _glossary(self) -> str:
        table = getattr(self.backend, "glossary", None) or {}
        if not table:
            return ""
        items = "；".join(f"{k}→{v}" for k, v in list(table.items())[:20])
        return f"- 术语表（必须按此译）：{items}\n"

    def _llm(self, text: str, src: str, tgt: str, strict_numbers: bool = False) -> str:
        prompt = PROMPT.format(
            src=src,
            tgt=tgt,
            glossary=self._glossary(),
            context=self._context(),
            text=text,
        )
        if strict_numbers:
            prompt += "\n注意：译文必须包含原句里的所有数字，一个都不能少。\n"
        if self.cfg.translator == "openai":
            return self.backend._openai(prompt)  # noqa: SLF001
        return self.backend._ollama(prompt)  # noqa: SLF001

    def _argos(self, text: str) -> str:
        protected, mapping = self._protect(text)
        raw = self.backend._argos(protected)  # noqa: SLF001
        return self._restore(raw, mapping)

    @staticmethod
    def _protect(text: str) -> tuple[str, dict[str, str]]:
        mapping: dict[str, str] = {}
        out = text
        for i, number in enumerate(numbers_of(text)):
            token = f"⟦{i}⟧"
            if number in out:
                out = out.replace(number, token, 1)
                mapping[token] = number
        return out, mapping

    @staticmethod
    def _restore(text: str, mapping: dict[str, str]) -> str:
        out = text
        for token, number in mapping.items():
            out = out.replace(token, number)
        return re.sub(r"[ \t]{2,}", " ", out).strip()
