# VoiceTwin · 声替

> 不是把话翻译成中文，而是让那个人**用他自己的声音**，继续说中文。

## 它和市面同传的「本质不同」

| 维度 | 传统同传 / 翻译机 | VoiceTwin |
|---|---|---|
| 产物 | 中文配音 / 字幕，播音腔 | **同一说话人用自己的音色说中文** |
| 翻译单位 | 逐句直译 | **译员式意译**：压缩冗余、补语境、保语气 |
| 声音 | 系统音色 | **原声克隆**（3–10 秒参考音频，可自动从源音频截取） |
| 多人会议 | 混成一轨 | **声纹分离**，每人一个声音（装 pyannote 后启用） |
| 时间轴 | 每句按自己的节奏播，越说越错位 | **压回原句时长**：静音裁剪 + 有限加速，不让漂移累积 |
| 来源 | 无从追溯 | **vtwn 溯源块**：引擎、参考音频指纹、授权来源直接写进音频文件 |
| 部署 | 云端 | **全部本机**，音频不出机器 |

一句话：**它翻译的不是语言，是「说话的人」**——翻译层消失，变成「同一个人换了种语言」。

## 架构

```
音频 → VAD 切句 → ASR 识别 → 说话人分离 → 译员（意译） → 音色克隆 TTS → 按句落盘
                                  ↑                ↓
                          术语表 / 风格       参考音频（自动截取或手给）
```

五个阶段各自是独立类（`voicetwin/stages.py`），都能用真实实现替换、也都有 mock 实现。
**克隆 TTS 通过 HTTP 契约接入**，所以 CosyVoice2 / GPT-SoVITS / 你自己的模型都能挂进来。

每句配音落盘后还有两道**本项目独有**的后处理：

- `voicetwin/timing.py` —— 时间轴对齐：按 10 ms 帧能量剪掉首尾静音，仍偏长就限速重采样（默认上限 1.3×，避免花栗鼠音），偏短就补静音，让每句严格占满原句时长。纯标准库，不需要 ffmpeg。
- `voicetwin/provenance.py` —— 音色溯源：往 wav 里写一个自有 RIFF 块 `vtwn`（`VTWN1\n` + JSON），记录引擎、参考音频 sha256、授权来源，**不改 audio data 一个字节**，任何播放器照常播。

```bash
python -m voicetwin run --transcript demo/transcript.tsv --out out --consent granted
# --no-fit 关掉时间轴对齐；--no-stamp 关掉溯源块
```

## 三种跑法

```bash
cd voice-twin

# 1) 零依赖：mock 链路（不需要任何模型）
python -m voicetwin run --transcript demo/transcript.tsv --out out
python -m voicetwin compare --transcript demo/transcript.tsv --out out   # 直译 vs 译员式

# 2) 浏览器演示台（真实 ASR，可拖音频）
python -m voicetwin serve --port 8765

# 3) 命令行真实链路（需要 .venv312，见下）
.venv312\Scripts\python -m voicetwin run --no-mock --audio demo/en_speech.wav \
    --translator argos --asr-model tiny.en --tts sapi --out out/real

# 4) 分块同传：按 3 秒切块，边切边译，量的是「延迟」不是「总耗时」
.venv312\Scripts\python -m voicetwin stream --audio demo/en_speech.wav --window 3.0 \
    --asr-model tiny.en --translator passthrough --out out/stream
```

`serve` 只监听 `127.0.0.1`：上传音频 → 跑完整链路 → 页面展示每阶段耗时、逐句译文、可播放配音、SRT 下载。
页面上把「TTS」选成 CosyVoice2 并填上克隆服务地址（默认 `http://127.0.0.1:9880/synthesize`），就是完整的换声链路。

## 本机实测（2026-11-11，CPU，无 GPU / 无 ffmpeg）

| 链路 | 命令 | 结果 |
|---|---|---|
| ASR | `--translator passthrough --asr-model tiny.en` | 7 秒音频 → 2 句，**4.4 s**（RTF≈0.64），文本与原句一致 |
| 离线翻译 | `--translator argos` | 真实中文输出：`实时翻译一直涉及文字.` / `但如果我们翻译了这个人呢?` |
| 克隆 TTS 契约 | `--tts http --tts-url http://127.0.0.1:9880/synthesize` | 每句落盘 `001.wav` / `002.wav`，参考音频自动截取 `reference.wav` |
| 时间轴对齐 | `fit_to_timeline` 对 3 s 音频对齐到 1 s | 产物严格 1.00 s，超出部分 1.3× 加速后淡出截断 |
| 溯源块 | `stamp` 后 `read_stamp` | 引擎 / consent / 参考音频 sha256 全部读回，audio data 字节不变 |
| 分块同传 | `stream --window 3.0 --asr-model tiny.en` | 3 块、每块 **约 460 ms** 出结果，RTF **0.20**，首块 465 ms，两句话完整无截断 |
| 测试 | `.venv312\Scripts\python -m unittest discover -s tests` | **33 passed** |

> 表中的 `argos` 是本机无 LLM 时的兜底：**质量明显弱于 LLM**，只证明链路真能出中文。
> 想要译员级质量，用 Ollama 或 OpenAI 兼容 API。

## 分块同传：把「说完多久出中文」量出来

整段跑完的总耗时对同传没有意义。`stream` 子命令按固定窗口切块、逐块识别+翻译，
每块都打印延迟与进度（落后=处理慢于音频，领先=快于音频）：

```
输入：demo/en_speech.wav  块数：3  窗口：3.0s
[   0.0s →    3.0s] 延迟    474 ms    领先 2526 ms  Real-time translation has always been about words.
[   3.0s →    6.0s] 延迟    443 ms    领先 5083 ms  But what if we translated the person instead?
[   6.0s →    6.8s] 延迟    450 ms    领先 5481 ms  （无语音）

RTF 0.20（可实时）  首块延迟 474 ms  全程领先 5481 ms
```

三条设计取舍（都写进了代码）：

- **模型只加载一次**：`Asr` 内部缓存 `WhisperModel`；首次加载走 `warmup()` 空跑静音，不计入延迟。
- **跨块接上下文**：上一块识别文本的尾部作为 `initial_prompt` 喂给下一块，句子不会在块边界被切碎。
- **静音不脑补**：`no_speech_prob > 0.6` 的块直接丢弃，否则静音尾巴会被写成 "Thank you."。

产物：`stream.md`（延迟表）、`stream.srt`（可直接挂播放器）。

> RTF < 1 只说明「追得上音频」，不等于端到端 1.5 s：3 s 块 + 0.47 s 处理 ≈ 3.5 s 端到端。
> 要压到 1.5 s 以内需要流式 ASR（边出边改）+ 更小的块，见 `docs/ARCHITECTURE.md` §14。

## 接真模型

> 系统 Python 是 3.15，ML 轮子还没覆盖（`ctranslate2` 无 cp315 wheel）；用 **3.12** 建 venv。

```bash
py -3.12 -m venv .venv312
.venv312\Scripts\python -m pip install -r requirements-real.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

翻译后端四选一：

| 后端 | 命令 | 说明 |
|---|---|---|
| Ollama（本地 LLM） | `--translator ollama --model qwen2.5:7b` | 需先 `ollama serve` |
| OpenAI 兼容 API | `--translator openai --api-base <url> --api-key <key>` | 也可用 `VOICETWIN_API_KEY` |
| Argos（离线 NMT） | `--translator argos`，先 `python -m voicetwin fetch-argos` | 无需 API key，质量一般 |
| 直通（调 ASR） | `--translator passthrough` | 只验证识别 |

### 换声（克隆 TTS）

没有 GPU 也能把接口跑通，参考实现带一个可测的桩服务：

```bash
# 终端 A：参考合成服务（tone 只为验证契约；sapi 用系统音色但受安全策略限制）
python scripts/cloning_server.py --engine tone --port 9880

# 终端 B：真实链路 + 换声
.venv312\Scripts\python -m voicetwin run --no-mock --audio demo/en_speech.wav \
    --translator argos --tts http --tts-url http://127.0.0.1:9880/synthesize --out out/clone
```

契约见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 10 节：

```
POST /synthesize
{"text": "...", "reference_audio": "/abs/ref.wav", "language": "zh"}
→ 200 audio/wav bytes
```

把 CosyVoice2 / GPT-SoVITS 的 api server 包一层，或在 `--tts-cmd` 里直接调它们的 CLI，都能接上。
**注意**：真正的克隆音色需要 GPU 上跑引擎，本仓库不含模型权重。

## 环境坑（已在代码里处理）

- **PyAV**：`faster-whisper 1.2.1` 需要 `av>=14,<15`，PyAV 19 删了 `metadata_errors` 参数。
- **HuggingFace**：默认 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=<项目>/.hf-cache`。
- **受限目录**：`~/.cache`、`~/.local/share`、`~/.local` 在本机写不进去 → Argos 的 XDG 目录、stanza 资源目录全部指到项目内 `.argos/`，在 import 前设置。
- **stanza 索引**：默认走 `raw.githubusercontent.com`，换成 jsDelivr 镜像，并打补丁容忍缺失 `content-length`。
- **SAPI**：`System.Speech` 在本机被安全策略拦截（能枚举音色、`Speak()` 失败、留下 0 字节文件），代码会显式报错而不是静默成功。

`python -m voicetwin doctor` 会逐项检查这些。

## 目录

```
voicetwin/
  config.py     # 语言对、模型名、TTS 端点、参考音频、对齐与溯源开关
  types.py      # Segment / Translated / Synthesis / RunReport
  stages.py     # VAD / ASR / 分离 / 译员 / TTS（mock + real）+ 后处理
  pipeline.py   # 串联、计时、单阶段失败降级
  timing.py     # 时间轴对齐（原创算法，纯标准库）
  provenance.py # vtwn 溯源块（原创 RIFF 格式）
  report.py     # Markdown 报告 + run.json
  server.py     # 零依赖 Web 演示台
  cli.py        # run / compare / serve / doctor / fetch-argos
scripts/
  cloning_server.py  # 克隆 TTS 的参考实现（HTTP 契约）
demo/           # 示例数据（en_speech.wav 为系统语音合成，用于 ASR 验收）
docs/           # 架构、延迟预算、选型、路线图
tests/          # 27 个单元测试，核心部分零依赖
```

## 原创边界（这条很重要）

本仓库**不是把别人的项目拼起来**：

| 属于本项目的原创部分 | 说明 |
|---|---|
| 整条管线与降级策略 | 五阶段串联、单阶段失败不拖垮全链路、逐阶段计时 |
| 时间轴对齐算法 | `timing.py` 全程自写，无 ffmpeg / 无第三方 DSP |
| `vtwn` 溯源块格式 | 自定 RIFF 块，读写实现自写，带 consent 语义 |
| 克隆 TTS HTTP 契约 | 请求/响应格式由本项目定义，任何引擎按此接入 |
| 译员式重述提示词 | 「压缩冗余、补语境、保语气」的 interpreter 风格 |
| Web 演示台 | 纯标准库 http.server，路由与安全策略自写 |
| 测试与实测数据 | 全部在本机跑出来，不是抄的 benchmark |

**外部只作为可选适配器**：ASR 用 faster-whisper、分离用 pyannote、离线翻译用 Argos、克隆引擎用 CosyVoice2/GPT-SoVITS——它们都通过 `pip`/HTTP 在运行时接入，仓库里**不含任何第三方源码或模型权重**，你换掉任何一个都不影响项目主体。

## 发布

```bash
pip install build && python -m build        # 产出 dist/voicetwin-0.0.1-*.whl
python -m twine upload dist/*              # 需要先配好 PyPI token
```

## 伦理红线

- 只克隆**已获同意**的声音；输出音频带 `voicetwin` 元数据标记。
- 全部本地运行，音频不出机器；`serve` 只绑定回环地址。
- 不做「冒充他人」用途。
