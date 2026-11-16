"""统一中间表示：CharUnit / Phrase / SongDoc。

严格对应技术实现文档第三节数据契约。先写本文件，再写其他模块。
- 时间单位一律秒（float），不混用毫秒或帧号
- 音频中间产物统一 44100 Hz wav
- F0 对齐到 10ms 帧移
- Phrase 是编辑与重生成的原子单位
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# 注意：文档示例代码 import numpy 但未实际使用。这里保留常量定义，
# 避免为未使用的依赖在契约测试里引入重型库。
F0_HOP_MS = 10
SR = 44100


class ContractError(Exception):
    """中间表示违反数据契约（时间重叠、零时长等）。"""


def validate_song_doc(doc: "SongDoc") -> None:
    """校验 SongDoc 满足 P0 契约：时间单调递增、句间不重叠、无零时长单元。"""

    prev_phrase_end = 0.0
    for phrase in doc.phrases:
        if phrase.end <= phrase.start:
            raise ContractError(f"第 {phrase.idx} 句时长非正：start={phrase.start}, end={phrase.end}")
        if phrase.start < prev_phrase_end:
            raise ContractError(
                f"第 {phrase.idx} 句与前一句重叠或时间未单调递增："
                f"start={phrase.start} < 前句 end={prev_phrase_end}"
            )
        if phrase.n_syl != len(phrase.chars):
            raise ContractError(f"第 {phrase.idx} 句 n_syl 与 chars 长度不一致")

        prev_char_end = None
        for char in phrase.chars:
            if char.end <= char.start:
                raise ContractError(
                    f"第 {phrase.idx} 句存在零时长字「{char.ch}」：start={char.start}, end={char.end}"
                )
            if prev_char_end is not None and char.start < prev_char_end:
                raise ContractError(
                    f"第 {phrase.idx} 句字级时间未单调递增：字「{char.ch}」start={char.start} "
                    f"小于前字 end={prev_char_end}"
                )
            prev_char_end = char.end
        prev_phrase_end = phrase.end


@dataclass
class CharUnit:
    """单个字的对齐与音高。f0 为逐帧基频，清音为 None。"""

    idx: int
    ch: str
    start: float
    end: float
    f0: Optional[list] = None
    hop_ms: int = F0_HOP_MS


@dataclass
class Phrase:
    """一句歌词，是编辑与重生成的原子单位。"""

    idx: int
    text: str
    start: float
    end: float
    chars: list = field(default_factory=list)
    new_text: Optional[str] = None
    edited: bool = False
    new_vocal_path: Optional[str] = None

    @property
    def n_syl(self) -> int:
        return len(self.chars)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SongDoc:
    """整首歌的中间表示，pipeline 唯一载体。可序列化为 song_doc.json。"""

    src_path: str
    vocal_path: str = ""
    accomp_path: str = ""
    sr: int = SR
    phrases: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "SongDoc":
        phrases = [
            Phrase(
                idx=p["idx"],
                text=p["text"],
                start=p["start"],
                end=p["end"],
                chars=[CharUnit(**c) for c in p.get("chars", [])],
                new_text=p.get("new_text"),
                edited=p.get("edited", False),
                new_vocal_path=p.get("new_vocal_path"),
            )
            for p in data.get("phrases", [])
        ]
        return cls(
            src_path=data["src_path"],
            vocal_path=data.get("vocal_path", ""),
            accomp_path=data.get("accomp_path", ""),
            sr=data.get("sr", SR),
            phrases=phrases,
        )

    @classmethod
    def from_json(cls, path: str) -> "SongDoc":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
