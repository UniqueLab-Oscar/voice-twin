# VoiceTwin 架构与选型

## 1. 为什么是「换声」而不是「翻译」

传统同传的链路是：

```
ASR → MT（逐句直译） → TTS（系统音色）
```

它把「说话的人」丢掉了：你听到的永远是同一个播音腔机器人在念译文，且逐句直译会丢上下文。

VoiceTwin 把 TTS 换成**零样本音色克隆**，并把 MT 换成**译员式重述**：

```
ASR → 说话人分离 → 译员 LLM（意译 + 压缩 + 语气） → 零样本克隆 TTS（原说话人音色）
```

结果：你听到的是「他本人在用中文跟你说话」，而不是「有人在翻译他」。

## 2. 延迟预算（目标 < 1.5s）

| 阶段 | 预算 | 手段 |
|---|---|---|
| VAD 切句 | 100 ms | silero-vad，300ms 滑窗 |
| ASR | 300 ms | faster-whisper small/medium，流式 chunk |
| 译员 LLM | 400 ms | 7B 量化本地模型，短输出、流式首 token |
| 克隆 TTS | 300 ms | CosyVoice2 流式合成 |
| 混音/路由 | 100 ms | 虚拟声卡 |

第二阶段「预测式」：边说边预测后半句，先出音频再校正，目标 500 ms。

## 3. 组件选型（2026-11 核实）

| 环节 | 首选 | 备选 | 许可证 | 备注 |
|---|---|---|---|---|
| VAD | `snakers4/silero-vad` 10.6k★ | — | MIT | 轻、准、CPU 可跑 |
| ASR | `SYSTRAN/faster-whisper` 26.6k★ | `modelscope/FunASR` 21.0k★、`k2-fsa/sherpa-onnx` 15.9k★ | MIT / MIT / Apache-2.0 | faster-whisper 最省事 |
| 说话人分离 | `pyannote/pyannote-audio` 10.8k★ | — | MIT | 需 HF token |
| 翻译/意译 | 本地 LLM（Ollama + Qwen2.5） | — | 模型各自许可 | 译员 prompt 是核心 |
| 克隆 TTS | `QwenAudio/CosyVoice` 24.4k★ | `RVC-Boss/GPT-SoVITS` 64.7k★ | **Apache-2.0** / MIT | CosyVoice2 跨语种零样本最稳 |
| （慎用） | `fishaudio/fish-speech` 33.7k★ | `facebookresearch/seamless_communication` 11.9k★ | **NOASSERTION** | 许可证非标准，商用前必须确认 |
| 音频路由 | VB-Cable（Win）/ BlackHole（mac） | — | 各自 | 把译文混进耳机 |

> 星数与推送时间取自 GitHub API 实测；`fish-speech`、`seamless_communication` 的 license 字段为 NOASSERTION，**不要默认可商用**。

## 4. 译员 Prompt（核心差异点）

不是「翻译下面这句话」，而是：

```
你是一名资深同声传译员。听者不懂源语言，只能听你的译文。
规则：
1. 以「让听者听懂」为第一目标，可压缩冗余、合并短句、补必要背景。
2. 保留说话人的语气、幽默、强调；不要书面化。
3. 术语按术语表统一；不确定的专有名词保留原文读音。
4. 输出只给译文，不要解释。
```

对比实验：直译 prompt vs 译员 prompt，用同一段演讲人工打分（准确 / 可听性 / 语气保留）。

## 5. 数据流与错误处理

- 每个阶段失败都不拖垮整条链路：ASR 失败 → 标记该句 `[未识别]`；TTS 失败 → 回退系统音色并告警。
- 所有中间产物落盘（`out/<n>.json`），可离线重放与调参。
- 报告 `out/report.md` 含原文、译文、说话人、耗时。

## 6. 路线图

- **W1**：mock 链路 + 真 ASR + 真翻译（先不做克隆，系统音色）
- **W2**：接 CosyVoice2 零样本克隆，单说话人端到端
- **W3**：pyannote 多人分离，每人一个音色；虚拟声卡路由
- **W4**：流式化 + 延迟压测 + 预测式实验

## 7. 风险

- 克隆音色的**稳定性**（长句漂移）→ 分句短、参考音频固定。
- 跨语种克隆的**口音残留**→ 选 CosyVoice2 跨语种模式，或用 GPT-SoVITS 微调。
- **延迟**是硬约束，GPU 是前提；CPU 只能离线。
- **伦理**：必须内置同意校验与水印，否则这项目不该开源。

## 8. 本机环境实测（2026-11-11）

| 项 | 实测 | 影响 |
|---|---|---|
| Python | 系统 3.15.2（`C:\Python315`），**已装 3.12.10**（`py -3.12`，路径 `%LOCALAPPDATA%\Programs\Python\Python312`） | 3.15 上 `pip install faster-whisper` 报 `ResolutionImpossible`：ctranslate2 没有 cp315 wheel。3.12 可用 |
| GPU | `nvidia-smi` 不存在，未检出 NVIDIA 显卡 | 7B 模型实时推理不可行，只能 CPU 离线或外接 |
| ffmpeg | 未安装 | 音频解码/重采样需先装 |
| 磁盘 | C 盘剩余 ~118 GB | 够装模型 |

**安装踩坑与可用路径（国内网络）**：

- `pip install uv` 在 3.15 上 `No matching distribution found for uv`；`winget install Python.Python.3.12` 下载 python.org 报 `InternetReadFile() failed. 0x80072ee2`。
- 可行：从华为镜像下载安装包再用静默参数装 per-user（无需管理员）：
  `https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe`
  `/quiet InstallAllUsers=0 PrependPath=0 Include_pip=1 Include_test=0`
- pip 用清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`
- HuggingFace 直连受限，下 whisper 模型前设 `HF_ENDPOINT=https://hf-mirror.com`。

**部署建议**：用 `py -3.12` 建 venv（本项目用 `.venv312`），不要用系统 3.15；实时模式需要一块 GPU（或把模型推理放到外接/云端）。

### 8.1 真实链路实测（2026-11-11）

已用 `.venv312`（Python 3.12.10）跑通「音频 → ASR → 译员(passthrough) → SRT」：

```
python -m voicetwin run --no-mock --audio demo/en_speech.wav --translator passthrough --asr-model tiny.en --out out/real
```

- 7.0 秒音频 → 2 句；`tiny.en` + CPU int8，ASR 4.46 s（RTF ≈ 0.64）；全链路 4.46 s。
- 识别结果与 SAPI 原句一致：`Real-time translation has always been about words.` / `But what if we translated the person instead?`
- 已装版本：faster-whisper 1.2.1、ctranslate2 4.9.0、av 14.2.0、onnxruntime 1.31.1、tokenizers 0.23.3。

两个必须记住的坑：

1. **PyAV 19 与 faster-whisper 1.2.1 不兼容**：`av.open() got an unexpected keyword argument 'metadata_errors'`。锁 `av>=14,<15`（requirements-real.txt / pyproject `[real]` 已写）。
2. **HF 缓存目录**：`C:\Users\DELL\.cache\huggingface` 在本机 `PermissionError`，所以代码里默认 `HF_HOME=<项目>/.hf-cache`。

未安装可选项（会打印警告但不中断）：silero-vad（VAD）、pyannote.audio（说话人分离，依赖 torch，CPU 太重）。

**SAPI 合成在本机失败**：`Speak()` 报「系统上未安装语音，或没有当前安全设置可用的语音」，产物是 0 字节 WAV；
但 `GetInstalledVoices()` 能看到 `Microsoft Huihui Desktop` / `Microsoft Zira Desktop` 且 `enabled=True`，
说明是进程安全策略拦截，不是缺语音包。因此 `VoiceTts._sapi` 增加了「产物为 0 字节即报错」的保护；
在能用的机器上它会正常出音频，但**不换声**，只用于验证「译文 → 音频」这一段。

## 9. 开发顺序（据环境调整）

1. **离线文件版**：给一段英文音频，输出中文字幕 + 中文配音轨（CPU 可跑，慢但能验证质量）。
2. **实时版**：流式 ASR + 流式 TTS，压延迟（需要 GPU）。
3. **换声版**：接 CosyVoice2 零样本克隆（需要 GPU + 参考音频）。

## 10. 克隆 TTS 的接入契约

VoiceTwin 不绑定具体克隆引擎，只认一个 HTTP 契约，因此 CPU 机器也能把整条链路跑通、把接口测掉：

```
POST {tts_url}
Content-Type: application/json
{"text": "译文", "reference_audio": "/abs/path/ref.wav", "language": "zh"}

200 OK
Content-Type: audio/wav
<wav 字节>
```

- 参考音频优先取「说话人分离」给出的逐句参考；没有则用 `--ref-audio`；再没有，管线自动从源音频截前 `ref_seconds`（默认 8 秒）生成 `reference.wav`——**一段音频就能自举出说话人音色**，不需要提前准备样本。
- 也可以用 `--tts-cmd "python cosy.py --text {text} --ref {ref} --out {out}"`，适合直接调 CLI 的引擎。
- 参考实现：`scripts/cloning_server.py`（`tone` / `sapi` / `cmd` 三种后端）。它证明契约可用，但**不产生克隆音色**。

真实引擎接法（都需要 GPU）：

- **CosyVoice2**（`QwenAudio/CosyVoice`，Apache-2.0）：`cosyvoice2` 自带 `webui.py` 与 API 示例，包一层返回 wav 即可。
- **GPT-SoVITS**（MIT）：自带 `api_v2.py`，同样是「文本 + 参考音频 → wav」。

## 11. Web 演示台

`python -m voicetwin serve --port 8765`：只用标准库 `http.server`，零依赖。

| 路由 | 用途 |
|---|---|
| `GET /` | 单页界面：拖拽音频、选后端、看逐句结果 |
| `POST /api/upload?name=` | 原始字节上传（不经 multipart），落 `out/web/uploads/` |
| `POST /api/run` | 同步跑管线，返回逐句、耗时、SRT 链接 |
| `GET /api/audio` / `GET /api/file` | 播放配音 / 下载字幕 |

安全约束：只监听 `127.0.0.1`；文件名取 basename 并做字符白名单；读文件时校验目标位于输出目录之下（防目录穿越，已有测试覆盖）。

## 12. 本轮实测补充（2026-11-11）

| 场景 | 结果 |
|---|---|
| 离线翻译 `--translator argos` | 真实中文输出；首次调用含 stanza 模型加载，约 18.8 s，之后明显更快 |
| 换声契约 `--tts http` | 每句落盘 `001.wav` / `002.wav`，`reference.wav` 自动生成（前 8 s） |
| Web 演示台 | `GET /` 200；`POST /api/run` 返回 2 句 + 各阶段耗时 |
| Web 全链路换声 | tiny.en + argos + `tts_url` → `rows=2`，`001.wav` 42282 B，`GET /api/audio` 200 |
| 只出字幕（`tts=none`） | 仍返回 2 行译文（`ok=False`、无音频），不会被误判成「没识别」 |
| 测试 | 14 passed（含 HTTP 契约、参考音频截取、目录穿越防护、字幕-only 保留） |

Argos 的价值是「无 API key 也能出中文」，用于验证链路；其 NMT 质量不足以替代 LLM（例：「一直涉及文字」）。

stanza 的两个网络坑（代码已处理）：模型索引默认走 `raw.githubusercontent.com`（换 jsDelivr 镜像），且其 `download_file` 强依赖 `content-length`（jsDelivr 分块响应没有该头，已打补丁改为流式写盘）。


## 13. 两个原创组件

### 13.1 时间轴对齐（`voicetwin/timing.py`）

问题：同一句话中文更短，克隆 TTS 按自己的节奏播，会让后续每一句都比原声晚，越说越漂。

做法（纯标准库，无 ffmpeg）：

1. 10 ms 一帧算 RMS，以本段峰值的 2% 为阈值剪掉首尾静音；
2. 仍超长 → 线性插值重采样加速，速度上限 `max_speed`（默认 1.3×），超出直接截断并做 10 ms 淡出；
3. 偏短 → 尾部补静音，使每句严格占满 `segment.end - segment.start`。

加速会整体升调，是有意的取舍：口语同传里「跟上」优先于「音高绝对不变」。

### 13.2 音色溯源块（`voicetwin/provenance.py`）

每个合成 wav 写入自定义 RIFF 块：

```
id: vtwn   size: uint32   payload: "VTWN1\n" + utf-8 JSON
```

字段：`engine`、`consent`（self / granted / public-domain）、`created`、`reference_sha256`、`reference_name`、`tool`、`format`。

关键约束：**不改动 audio data 一个字节**，写块后仍能被标准库 `wave` 正常解析；块插在 `fmt ` 之后，重复写入只替换不重复；文件用临时文件 + `os.replace` 原子落盘。

`verify_license_notice(path, sha)` 可反查「这段配音是否由这段参考音频生成」，是伦理红线在代码里的落点。

## 14. 分块同传与延迟模型（2026-11-14）

### 14.1 三个指标
- `latency_ms`：一块音频从开始识别到译文可用的耗时（同一台机器、同一进程）；
- `lag_ms`：该块处理完时，`处理墙钟时间 - 已消费音频时长`；正数=落后（积压），负数=领先；
- `rtf`：总处理耗时 / 音频总时长，< 1 才有实时追赶能力。

### 14.2 切块策略
`stream.split_wav` 用纯标准库 `wave` 按固定窗口切，不依赖 ffmpeg：

- 边界先算好再写盘，避免切出空块；
- 最后一块不足 `MIN_CHUNK_SECONDS = 0.4s` 时并进上一块；
- 块内时间在 `StreamRunner` 里加回块起点，落到整段音频的绝对时间轴。

### 14.3 质量保命措施
| 问题 | 措施 |
|---|---|
| 每块重复加载模型 | `Asr` 缓存 `WhisperModel` 实例 |
| 首块延迟被模型加载污染 | `Asr.warmup()` 空跑 0.3s 静音，成本不计入统计 |
| 块边界把句子切断 | 上一块文本尾部（≤240 字符）作为下一块 `initial_prompt` |
| 静音块被脑补成 "Thank you." | `no_speech_prob > 0.6` 的 segment 丢弃 |

### 14.4 本机实测（tiny.en，CPU，窗口 3.0s）
```
[0.0-3.0s] 延迟 474 ms  领先 2526 ms  Real-time translation has always been about words.
[3.0-6.0s] 延迟 443 ms  领先 5083 ms  But what if we translated the person instead?
[6.0-6.8s] 延迟 450 ms  领先 5481 ms  （无语音）
RTF 0.20
```
两句话完整、无截断、无幻觉；每块处理约 0.45 s，远快于 3 s 音频。

### 14.5 离「端到端 1.5 s」还差什么
当前端到端 ≈ 块长 + 处理 + 播放缓冲 ≈ 3.0 + 0.45 + 缓冲，落在 3.5 s 量级。
要压到 1.5 s 以内，按收益排序：
1. 流式 ASR（partial hypothesis，边出边改）——0.8 s 就能出第一版字；
2. 块长降到 1.0–1.2 s，用 VAD 在停顿处提前封块；
3. 翻译走本地小模型或缓存命中，避免网络往返；
4. 克隆 TTS 先出首页音频、边合成边播（chunked streaming）。

### 14.6 测试
`tests/test_stream.py` 6 个用例覆盖：均分切块、短尾巴合并、非法窗口、块内时间偏移、
空块无译文、SRT/报告渲染。全程不加载模型。
