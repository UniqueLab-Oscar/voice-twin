# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## 0.0.1 — v0.0.1-snapshot（2026-11-12）

第一个可发布快照。核心命题：**同传翻译的不是「话」，是「人」**——原说话人用自己的音色继续说你的语言。

### 新增

- **五段式管线**：VAD → ASR（faster-whisper）→ 说话人分离（pyannote/silero）→ 译员式 LLM 重述 → 零样本克隆 TTS。
- **时间轴对齐**（`voicetwin/timing.py`，原创）：静音裁剪 + 有限加速 + 尾部补静音，把每句配音压回原句时长，解决「越翻越漂」。
- **音色溯源块 vtwn**（`voicetwin/provenance.py`，原创 RIFF 块）：每段合成音频写入引擎、参考音频 sha256、同意来源；不改动 audio data 一个字节，任何播放器可正常播放。
- **克隆 TTS HTTP 契约** + 参考桩服务（`scripts/cloning_server.py`）：没有 GPU 也能端到端验证换声链路。
- **Web 演示台**（`voicetwin/server.py`，纯标准库）：拖拽上传 → 逐句译文 / 阶段耗时 / 配音播放 / SRT 下载，只监听 `127.0.0.1`。
- **离线翻译**：Argos 后端，无 API key 也能出中文；有 key 时接 Ollama 或任意 OpenAI 兼容端点。
- **术语表**：`.json` 或 `源=译` 文本，强制覆盖专有名词。
- **直译 vs 译员式对比**：同一条音频跑两遍，并排给出差异。

### 验证

- 真实链路实测：7 秒英文音频 → 2 句中文，CPU 上 ASR 4.4 s（RTF≈0.64），克隆配音落盘 `001.wav`/`002.wav`。
- 27 个单元测试全绿（含 HTTP 契约、参考音频截取、目录穿越防护、时间轴对齐、RIFF 块读写）。
- 详见 [README 实测表](README.md) 与 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

### 已知限制

- 克隆引擎（CosyVoice2 / GPT-SoVITS）需要 GPU，本仓库只定义契约，不内置模型权重。
- 说话人分离在 CPU 上很慢，仅建议离线文件版使用。
- Argos 译文质量明显弱于 LLM，定位是「无 key 时的兜底」。
