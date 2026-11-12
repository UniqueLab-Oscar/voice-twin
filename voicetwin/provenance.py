"""音频来源与同意声明：VoiceTwin 自有的 RIFF 元数据块。

换声这件事的伦理风险不在模型，而在「这段音频是不是本人同意的」。
所以每一段合成音频都写一个自定义 RIFF 块（id 为 ``vtwn``），内容是可读 JSON：

    VTWN1\\n{"engine": "cosyvoice2", "reference_sha256": "...", "consent": "self", ...}

这是本项目的原创格式：不依赖 ID3（wav 不支持）、不依赖第三方库，
并且**不改变 audio data 的任何一个字节**，所以任何播放器都能正常播。

格式：
    chunk id   : b"vtwn"
    chunk size : little-endian uint32
    payload    : b"VTWN1\\n" + utf-8 JSON
    padding    : 奇数长度补一个 \\0
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import wave
from pathlib import Path

MAGIC = b"VTWN1\n"
CHUNK_ID = b"vtwn"
VALID_CONSENT = {"self", "granted", "public-domain"}


def fingerprint(path: str | Path) -> str:
    """参考音频的 sha256，用来证明「这个音色来自哪段录音」。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_meta(
    engine: str,
    reference: str | Path | None = None,
    consent: str = "self",
    note: str = "",
) -> dict:
    if consent not in VALID_CONSENT:
        raise ValueError(f"consent 必须是 {sorted(VALID_CONSENT)} 之一，收到 {consent!r}")
    meta: dict = {
        "engine": engine,
        "consent": consent,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "voicetwin",
        "format": "VTWN1",
    }
    if reference:
        ref_path = Path(reference)
        meta["reference_name"] = ref_path.name
        if ref_path.is_file():
            meta["reference_sha256"] = fingerprint(ref_path)
    if note:
        meta["note"] = note
    return meta


def _split_chunks(data: bytes) -> tuple[bytes, list[tuple[bytes, bytes]]]:
    """返回 (RIFF 头 12 字节, [(chunk_id, payload)])。"""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("不是合法的 RIFF/WAVE 文件")
    chunks: list[tuple[bytes, bytes]] = []
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        payload = data[pos + 8 : pos + 8 + size]
        chunks.append((cid, payload))
        pos += 8 + size + (size & 1)  # 奇数长度有 1 字节对齐填充
    return data[:12], chunks


def _assemble(header: bytes, chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = b"".join(
        cid + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")
        for cid, payload in chunks
    )
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body


def read_stamp(path: str | Path) -> dict | None:
    """读出 vtwn 块；没有或损坏返回 None。"""
    data = Path(path).read_bytes()
    try:
        _, chunks = _split_chunks(data)
    except ValueError:
        return None
    for cid, payload in chunks:
        if cid == CHUNK_ID and payload.startswith(MAGIC):
            try:
                return json.loads(payload[len(MAGIC) :].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
    return None


def stamp(path: str | Path, meta: dict) -> Path:
    """写入（或替换）vtwn 块。data 块字节保持不变，其他块原样保留。"""
    path = Path(path)
    data_bytes = _split_chunks(path.read_bytes())
    header, chunks = data_bytes
    payload = MAGIC + json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    kept = [(cid, body) for cid, body in chunks if cid != CHUNK_ID]

    # 插到 fmt 之后（紧跟格式描述，播放器与解析器都能尽早看到）
    out: list[tuple[bytes, bytes]] = []
    inserted = False
    for cid, body in kept:
        out.append((cid, body))
        if cid == b"fmt " and not inserted:
            out.append((CHUNK_ID, payload))
            inserted = True
    if not inserted:
        out.insert(0, (CHUNK_ID, payload))

    blob = _assemble(header, out)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)  # 原子替换，避免半截文件
    return path


def verify_license_notice(path: str | Path, expected_sha: str) -> bool:
    """校验「这段音频确实由这段参考音频生成」。"""
    meta = read_stamp(path)
    return bool(meta and meta.get("reference_sha256") == expected_sha)


def assert_wav_playable(path: str | Path) -> None:
    """写块之后仍然能被标准库正常解析——防止手写 RIFF 把文件写坏。"""
    with wave.open(str(path), "rb") as w:
        if w.getnframes() <= 0:
            raise ValueError("音频为空")
