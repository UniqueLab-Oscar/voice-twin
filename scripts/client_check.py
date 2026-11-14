"""客户端联调自检：不用打开浏览器，直接按前端的方式打 Web 演示台的每个接口。

演练的是浏览器里的真实顺序：
  1. GET /                      页面可打开
  2. POST /api/upload           拖进来的音频落到 uploads/
  3. POST /api/run              跑完链路，返回逐句结果与各阶段耗时
  4. GET  /api/file             SRT 能下载，且内容是合法字幕
  5. 越界与坏参数               必须被拒绝，而不是读到别的目录

用法：
    python -m voicetwin serve --port 8765 --no-open     # 终端 A
    python scripts/client_check.py --audio demo/en_speech.wav --asr-model tiny.en
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.status, r.read()


def _post(url: str, data: bytes, ctype: str) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:  # 4xx/5xx 也是被测行为
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VoiceTwin Web 演示台客户端自检")
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--audio", default="demo/en_speech.wav")
    ap.add_argument("--asr-model", default="tiny.en")
    ap.add_argument("--translator", default="passthrough", choices=["passthrough", "argos", "ollama", "openai"])
    ap.add_argument("--tts", default="none", choices=["none", "http", "cosyvoice2", "gpt-sovits"])
    ap.add_argument("--tts-url", default="")
    args = ap.parse_args(argv)

    base = args.base.rstrip("/")
    audio = Path(args.audio)
    if not audio.is_file():
        print(f"音频不存在：{audio}", file=sys.stderr)
        return 2

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'✓' if ok else '✗'} {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    status, body = _get(base + "/")
    check("GET /", status == 200 and b"VoiceTwin" in body, f"{status} / {len(body)} B")

    status, j = _post(
        base + "/api/upload?name=" + urllib.parse.quote(audio.name),
        audio.read_bytes(),
        "application/octet-stream",
    )
    uploaded = j.get("path", "")
    check("POST /api/upload", status == 200 and j.get("ok") and Path(uploaded).is_file(), uploaded)

    payload = json.dumps(
        {
            "audio": uploaded,
            "asr_model": args.asr_model,
            "translator": args.translator,
            "tts": args.tts,
            "tts_url": args.tts_url,
        }
    ).encode("utf-8")
    status, run = _post(base + "/api/run", payload, "application/json")
    rows = run.get("rows") or []
    check("POST /api/run", status == 200 and run.get("ok") and bool(rows), f"{len(rows)} 句")
    for r in rows:
        print(f"    [{r['i']}] {r['start']:.2f}s {r['speaker']}  {r['translated']}")
    print(f"    耗时：{run.get('timings')}")

    out = run.get("out", "")
    q = urllib.parse.urlencode({"name": "report.srt", "dir": out})
    status, srt = _get(base + f"/api/file?{q}")
    text = srt.decode("utf-8", "replace")
    check("GET /api/file (SRT)", status == 200 and "-->" in text, f"{len(srt)} B")

    bad = urllib.parse.urlencode({"name": "report.srt", "dir": "C:/Windows"})
    try:
        status, _ = _get(base + f"/api/file?{bad}")
        check("目录越界被拒绝", status == 403, f"{status}")
    except urllib.error.HTTPError as exc:
        check("目录越界被拒绝", exc.code == 403, f"{exc.code}")

    print()
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
        return 1
    print("全部通过：浏览器里点一遍该走的接口，这里都走通了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
