"""纯 onnxruntime 的 Marian 翻译（不依赖 torch / optimum）。

Xenova 导出的 opus-mt 是 encoder + decoder_model_merged（带 past_key_values）。
merged 图有个 ``use_cache_branch`` 开关：第一步喂空 past、置 False；
之后每步只喂 1 个 token、置 True，并把上一步的 present.* 当 past 传回去。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np


class OnnxMarian:
    def __init__(self, model_dir: str | Path, threads: int = 4, max_new_tokens: int = 96) -> None:
        self.dir = Path(model_dir)
        self.onnx = self.dir / "onnx"
        self.threads = threads
        self.max_new_tokens = max_new_tokens
        self._tok = None
        self._enc = None
        self._lock = threading.Lock()
        self._dec = None
        self._dec_past = None
        self._cfg: dict = {}
        self._enc_io = None
        self._dec_in = None
        self._dec_out = None

    # ---------- 加载 ----------

    def _load(self) -> None:
        if self._dec is not None:
            return
        import json

        import onnxruntime as ort
        from transformers import MarianTokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, self.threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._enc = ort.InferenceSession(str(self.onnx / "encoder_model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._dec = ort.InferenceSession(str(self.onnx / "decoder_model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._dec_past = ort.InferenceSession(
            str(self.onnx / "decoder_with_past_model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._tok = MarianTokenizer.from_pretrained(str(self.dir))
        self._cfg = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self._past_out = [o.name for o in self._dec_past.get_outputs()]

    def translate(self, text: str, src: str = "", tgt: str = "") -> str:
        with self._lock:
            return self._translate_locked(text)

    def _translate_locked(self, text: str, src: str = "", tgt: str = "") -> str:
        self._load()
        assert self._dec is not None and self._tok is not None
        text = (text or "").strip()
        if not text:
            return ""

        enc = self._tok([text], return_tensors="np")
        input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        mask = np.asarray(enc["attention_mask"], dtype=np.int64)

        hidden = self._enc.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]
        start = int(self._cfg.get("decoder_start_token_id", 65000))
        eos = int(self._cfg.get("eos_token_id", 0))

        out = self._dec.run(
            None,
            {
                "input_ids": np.array([[start]], dtype=np.int64),
                "encoder_attention_mask": mask,
                "encoder_hidden_states": hidden,
            },
        )
        names0 = [o.name for o in self._dec.get_outputs()][1:]
        logits = out[0]
        present = dict(zip(names0, out[1:]))
        enc_past = {n: t for n, t in present.items() if ".encoder." in n}
        dec_past = {n: t for n, t in present.items() if ".decoder." in n}
        tokens: list[int] = []
        nxt = int(np.argmax(logits[0, -1]))
        for _ in range(self.max_new_tokens):
            if nxt == eos:
                break
            tokens.append(nxt)
            feed = {
                "input_ids": np.array([[nxt]], dtype=np.int64),
                "encoder_attention_mask": mask,
            }
            feed.update({n.replace("present.", "past_key_values."): t for n, t in enc_past.items()})
            feed.update({n.replace("present.", "past_key_values."): t for n, t in dec_past.items()})
            out = self._dec_past.run(None, feed)
            logits = out[0]
            names1 = [o.name for o in self._dec_past.get_outputs()][1:]
            dec_past = dict(zip(names1, out[1:]))
            nxt = int(np.argmax(logits[0, -1]))
        return self._tok.decode(tokens, skip_special_tokens=True).strip()


_CACHE: dict[str, OnnxMarian] = {}


def translate(model_dir: str | Path, text: str, src: str = "", tgt: str = "", threads: int = 4) -> str:
    key = str(Path(model_dir).resolve())
    inst = _CACHE.get(key)
    if inst is None:
        inst = OnnxMarian(key, threads=threads)
        _CACHE[key] = inst
    return inst.translate(text, src, tgt)


if __name__ == "__main__":  # 手动冒烟
    import sys

    m = OnnxMarian(sys.argv[1] if len(sys.argv) > 1 else "models/mt-onnx-zh-en")
    for s in ["今天下午三点开会", "这个季度的目标是提升响应速度"]:
        t0 = time.time()
        print(f"{(time.time() - t0) * 1000:.0f}ms {m.translate(s)!r}")
