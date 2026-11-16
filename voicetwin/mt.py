"""本地神经翻译：opus-mt（CTranslate2 int8，CPU 上约 0.3–0.6 秒一句）。

优先用 CT2 模型（``models/mt-zh-en``、``models/mt-en-zh``），没有就退回 transformers。
两者都不需要联网，也不需要 API key。

注意：opus-mt 的 CT2 转换把 decoder 起始 token 写成了 ``</s>``，会让模型复读；
这里在转换后把 ``config.json`` 的 ``decoder_start_token`` 改成 ``<pad>``，
并在输出上做一次「周期复读截断」，实测能把复读消掉。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_DIRS = {
    ("zh", "en"): "mt-zh-en",
    ("en", "zh"): "mt-en-zh",
}
_HF = {
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
}


def _dedupe(tokens: list[str]) -> list[str]:
    """截掉模型周期性复读的尾巴：ABAB… -> A，AAAA… -> A。"""
    n = len(tokens)
    for p in range(1, n // 2 + 1):
        if n >= 2 * p and tokens[p : 2 * p] == tokens[:p]:
            return tokens[:p]
    return tokens


class OnnxMt:
    """ONNX Runtime 版 opus-mt（Xenova 导出，量化），比 torch 快一个量级。"""

    def __init__(self, model_dir: str | Path, threads: int = 4) -> None:
        self.dir = Path(model_dir)
        self.threads = threads
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer

        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"optimum.onnxruntime 不可用：{exc}") from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, self.threads)
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._tok = AutoTokenizer.from_pretrained(str(self.dir))
        self._model = ORTModelForSeq2SeqLM.from_pretrained(str(self.dir), session_options=opts)

    def translate(self, text: str, src: str, tgt: str) -> str:
        self._load()
        batch = self._tok([text], return_tensors="pt")
        out = self._model.generate(
            **batch, max_new_tokens=max(16, min(64, len(text) * 3)), num_beams=1, do_sample=False
        )
        return self._tok.batch_decode(out, skip_special_tokens=True)[0].strip()


_ONNX_CACHE: dict = {}


def translate_onnx(text: str, src: str, tgt: str, threads: int = 4) -> str:
    where = onnx_dir()
    if where is None:
        raise RuntimeError("没找到 models/mt-onnx-zh-en")
    key = str(where)
    inst = _ONNX_CACHE.get(key)
    if inst is None:
        inst = OnnxMt(where, threads=threads)
        _ONNX_CACHE[key] = inst
    return inst.translate(text, src, tgt)


class MarianMt:
    """按方向懒加载；CT2 优先，transformers 兜底。"""

    def __init__(self, cache_dir: str | Path | None = None, threads: int = 8) -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        if cache_dir:
            os.environ.setdefault("HF_HOME", str(cache_dir))
        self.threads = threads
        from .paths import app_root

        self.root = app_root() / "models"
        self._ct2: dict[tuple[str, str], tuple[object, object, object]] = {}
        self._hf: dict[tuple[str, str], tuple[object, object]] = {}
        self._lock = threading.Lock()

    # ---------- CT2 ----------

    def _ct2_path(self, key: tuple[str, str]) -> Path | None:
        name = _DIRS.get(key)
        if not name:
            return None
        path = self.root / name
        return path if (path / "model.bin").exists() else None

    def _load_ct2(self, key: tuple[str, str]):
        import ctranslate2
        import sentencepiece as spm

        path = self._ct2_path(key)
        if path is None:
            raise FileNotFoundError(key)
        translator = ctranslate2.Translator(
            str(path), device="cpu", compute_type="int8", inter_threads=1, intra_threads=self.threads
        )
        src_sp = spm.SentencePieceProcessor(model_file=str(path / "source.spm"))
        tgt_sp = spm.SentencePieceProcessor(model_file=str(path / "target.spm"))
        with self._lock:
            self._ct2[key] = (translator, src_sp, tgt_sp)
        return self._ct2[key]

    # ---------- transformers ----------

    def _load_hf(self, key: tuple[str, str]):
        from transformers import MarianMTModel, MarianTokenizer

        name = _HF[key]
        tok = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name)
        model.eval()
        with self._lock:
            self._hf[key] = (tok, model)
        return self._hf[key]

    # ---------- 对外 ----------

    def supports(self, src: str, tgt: str) -> bool:
        key = (src[:2], tgt[:2])
        return self._ct2_path(key) is not None or self._ct2_path((key[1], key[0])) is not None

    def translate(self, text: str, src: str, tgt: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        src, tgt = src[:2], tgt[:2]
        if src == tgt:
            return text
        key = (src, tgt)
        if not self._ct2_path(key):
            key = (tgt, src)  # 只装了一个方向时反向尝试
            if not self._ct2_path(key):
                return self._hf_translate(text, (src, tgt))

        if key not in self._ct2:
            self._load_ct2(key)
        translator, src_sp, tgt_sp = self._ct2[key]
        tokens = src_sp.encode(text, out_type=str)
        result = translator.translate_batch([tokens], beam_size=1, max_decoding_length=96)[0]
        hyp = _dedupe(list(result.hypotheses[0]))
        return tgt_sp.decode(hyp).strip()

    def _hf_translate(self, text: str, key: tuple[str, str]) -> str:
        if key not in self._hf:
            self._load_hf(key)
        tok, model = self._hf[key]
        import torch

        with torch.inference_mode():
            batch = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
            out = model.generate(**batch, max_new_tokens=max(16, min(96, len(text) * 3)), num_beams=1)
            return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    def warm(self, src: str = "zh", tgt: str = "en") -> None:
        key = (src[:2], tgt[:2])
        if self._ct2_path(key):
            self._load_ct2(key)
        else:
            self._load_hf(key)


def onnx_dir() -> Path | None:
    from .paths import app_root

    root = app_root() / "models"
    for name in ("mt-onnx-zh-en",):
        path = root / name
        if (path / "onnx" / "encoder_model.onnx").exists():
            return path
    return None


def available() -> bool:
    root = Path(__file__).resolve().parent.parent / "models"
    if (root / "mt-zh-en" / "model.bin").exists():
        return True
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def patch_decoder_start(model_dir: str | Path) -> None:
    """把 CT2 config 的 decoder_start_token 从 </s> 改成 <pad>（转换脚本的老毛病）。"""
    path = Path(model_dir)
    vocab_file = path / "shared_vocabulary.json"
    cfg_file = path / "config.json"
    if not vocab_file.exists() or not cfg_file.exists():
        return
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    if "<pad>" not in vocab:
        vocab.append("<pad>")
        vocab_file.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    if cfg.get("decoder_start_token") != "<pad>":
        cfg["decoder_start_token"] = "<pad>"
        cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
