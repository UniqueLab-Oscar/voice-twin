# 参与贡献

## 快速开始

```bash
git clone <your-fork-url> && cd voice-twin
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m unittest discover -s tests -v         # 核心测试零依赖，秒级跑完
```

真实链路需要额外依赖：

```bash
pip install -e ".[real]"      # faster-whisper / pyannote / silero
pip install -e ".[argos]"     # 离线翻译（可选）
```

## 提 PR 前

1. 跑完 `python -m unittest discover -s tests`，不要引入失败；
2. 新增能力请配测试，尤其是涉及音频字节改写的部分（时间轴、溯源块）；
3. 保持「核心零依赖」——只用标准库的模块不要悄悄加第三方 import；
4. 涉及换声/克隆的改动，必须说明数据授权来源，不得提交他人未授权的音色样本。

## 我们特别欢迎的方向

- 新的克隆引擎适配器（只实现 HTTP 契约即可，见 `docs/ARCHITECTURE.md` §10）；
- 更好的时间轴策略：按语义停顿对齐、多说话人并发；
- 流式（实时）管线：现在只做离线文件版；
- 语言对扩展：管线与语言无关，主要补 ASR / 翻译 / 音色三处的适配。

## 行为准则

不伪造音色归属，不把「同意」写成默认值糊弄过去。溯源块里的 `consent` 字段必须反映真实授权情况。
