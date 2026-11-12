from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    """一句（或一个语义块）的识别结果。"""

    start: float
    end: float
    text: str
    speaker: str = "S0"
    ref_audio: str = ""  # 该说话人的参考音频（用于音色克隆）


@dataclass
class Translated:
    """译员式重述后的结果。"""

    segment: Segment
    text: str
    style: str = "interpreter"


@dataclass
class Synthesis:
    """合成产物。"""

    translated: Translated
    audio_path: str
    voice: str
    ok: bool = True


@dataclass
class RunReport:
    segments: list[Segment] = field(default_factory=list)
    translations: list[Translated] = field(default_factory=list)
    syntheses: list[Synthesis] = field(default_factory=list)
    elapsed_ms: dict[str, float] = field(default_factory=dict)
