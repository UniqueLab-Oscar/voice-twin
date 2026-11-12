from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    src_lang: str = "en"
    tgt_lang: str = "zh"
    mock: bool = True

    # 译员 LLM（Ollama）
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b"
    translator: str = "ollama"  # ollama | openai | passthrough（调试 ASR 用）
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    style: str = "interpreter"  # interpreter | literal
    glossary: str = ""  # 术语表文件：.json {源:译} 或 .txt 每行 源=译

    # ASR
    asr_model: str = "small"
    asr_device: str = "auto"

    # 克隆 TTS
    tts_engine: str = "cosyvoice2"  # cosyvoice2 | gpt-sovits | http | sapi | system
    tts_cmd: str = ""  # 可选：外部合成命令，{text} {ref} {out} 占位符
    tts_url: str = ""  # 可选：克隆引擎的 HTTP 端点（见 docs/ARCHITECTURE.md 的契约）
    ref_audio: str = ""  # 说话人参考音频；留空则自动从源音频截取前几秒
    ref_seconds: float = 8.0  # 自动截取的参考音频长度
    sapi_voice: str = ""  # sapi 引擎的音色名，留空用系统默认（中文可用 "Microsoft Yaoyao"）

    # 时间轴与溯源（VoiceTwin 自有能力）
    fit_timing: bool = True  # 合成音频对齐到原句时长（见 timing.py）
    max_speed: float = 1.3  # 对齐时允许的最大加速倍率，避免变声
    consent: str = "self"  # 音色授权来源：self | granted | public-domain
    stamp: bool = True  # 是否给每条合成音频写 vtwn 溯源块（见 provenance.py）

    # 输出
    out_dir: str = "out"
