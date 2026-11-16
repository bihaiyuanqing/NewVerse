"""P0.4 数据契约与 splice 边界测试。

断言覆盖（文档第五节（一）P0.4）：
- 时间单调、句间不重叠
- n_syl 等于 chars 长度
- 无零时长单元
- splice 边界不越界
"""

import numpy as np
import pytest

from core.audio_io import CROSSFADE_MS, rms_db, splice
from core.errors import SpliceError
from core.schema import (
    SR,
    CharUnit,
    ContractError,
    Phrase,
    SongDoc,
    validate_song_doc,
)


def make_char(idx: int, ch: str, start: float, dur: float = 0.3) -> CharUnit:
    return CharUnit(idx=idx, ch=ch, start=start, end=start + dur)


def make_phrase(idx: int, text: str, start: float, dur: float = 0.3) -> Phrase:
    chars = [
        make_char(i, ch, start + i * dur, dur)
        for i, ch in enumerate(text)
    ]
    return Phrase(
        idx=idx,
        text=text,
        start=start,
        end=start + dur * len(text),
        chars=chars,
    )


def make_doc() -> SongDoc:
    p0 = make_phrase(0, "明天你好", 0.0)
    p1 = make_phrase(1, "声音多美妙", 2.0)
    return SongDoc(
        src_path="assets/demo/sample.mp3",
        vocal_path="output/vocals.wav",
        accomp_path="output/no_vocals.wav",
        phrases=[p0, p1],
    )


# ---------- n_syl 契约 ----------

def test_n_syl_equals_chars_length():
    phrase = make_phrase(0, "明天你好", 0.0)
    assert phrase.n_syl == len(phrase.chars) == 4


# ---------- 时间单调 / 不重叠 / 零时长 ----------

def test_valid_doc_passes_contract():
    validate_song_doc(make_doc())


def test_zero_duration_char_rejected():
    doc = make_doc()
    doc.phrases[0].chars[1].end = doc.phrases[0].chars[1].start
    with pytest.raises(ContractError):
        validate_song_doc(doc)


def test_zero_duration_phrase_rejected():
    doc = make_doc()
    doc.phrases[0].end = doc.phrases[0].start
    with pytest.raises(ContractError):
        validate_song_doc(doc)


def test_char_time_non_monotonic_rejected():
    doc = make_doc()
    doc.phrases[0].chars[2].start = doc.phrases[0].chars[0].start
    with pytest.raises(ContractError):
        validate_song_doc(doc)


def test_phrase_overlap_rejected():
    doc = make_doc()
    doc.phrases[1].start = 0.5  # 早于第 0 句结尾
    with pytest.raises(ContractError):
        validate_song_doc(doc)


# ---------- SongDoc 序列化 ----------

def test_song_doc_json_roundtrip(tmp_path):
    doc = make_doc()
    path = tmp_path / "song_doc.json"
    doc.to_json(str(path))
    restored = SongDoc.from_json(str(path))
    validate_song_doc(restored)
    assert restored.phrases[1].text == "声音多美妙"
    assert restored.phrases[1].n_syl == 5
    assert restored.phrases[0].chars[2].ch == "你"
    assert restored.sr == SR


# ---------- splice 边界与 50ms 交叉淡化硬约束 ----------

def test_splice_out_of_range_raises():
    sr = SR
    base = np.zeros(sr * 2, dtype=np.float32)  # 2 秒
    repl = np.ones(sr // 2, dtype=np.float32)  # 0.5 秒
    with pytest.raises(SpliceError):
        splice(base, repl, sr, -0.1, 0.4)  # 起点为负
    with pytest.raises(SpliceError):
        splice(base, repl, sr, 1.8, 2.4)  # 终点越界
    with pytest.raises(SpliceError):
        splice(base, repl, sr, 0.8, 0.7)  # end <= start


def test_splice_rejects_short_fade():
    sr = SR
    base = np.zeros(sr, dtype=np.float32)
    repl = np.ones(sr // 2, dtype=np.float32)
    with pytest.raises(SpliceError):
        splice(base, repl, sr, 0.0, 0.5, fade_ms=20)  # 短于 50ms 硬约束


def test_splice_length_mismatch_raises():
    sr = SR
    base = np.zeros(sr, dtype=np.float32)
    repl = np.ones(100, dtype=np.float32)
    with pytest.raises(SpliceError):
        splice(base, repl, sr, 0.0, 0.5)


def test_splice_crossfade_exact_50ms_and_blend():
    sr = SR
    base = np.zeros(sr, dtype=np.float32)       # 全零原声
    repl = np.ones(sr // 2, dtype=np.float32)   # 0.5s 全 1 新声
    out = splice(base, repl, sr, 0.0, 0.5)

    fade_n = int(CROSSFADE_MS / 1000.0 * sr)  # 2205 个采样
    assert fade_n == 2205
    assert out.shape == base.shape
    # 头部淡入：线性 0 -> 1
    np.testing.assert_allclose(out[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[fade_n - 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[fade_n // 2], 0.5, atol=1e-2)
    # 中段被完全替换为新声
    np.testing.assert_allclose(out[fade_n:sr // 2 - fade_n], 1.0, atol=1e-6)
    # 尾部淡出：最后一个淡化点应接近 0（原声为 0）
    np.testing.assert_allclose(out[sr // 2 - 1], 0.0, atol=1e-6)
    # 区间外保持原声不变
    np.testing.assert_allclose(out[sr // 2:], 0.0, atol=1e-6)


def test_splice_stereo_channel_aligned():
    sr = SR
    base = np.zeros((2, sr), dtype=np.float32)
    repl = np.ones((2, sr // 2), dtype=np.float32)
    out = splice(base, repl, sr, 0.0, 0.5)
    assert out.shape == base.shape
    np.testing.assert_array_equal(out[0], out[1])


def test_rms_db_detects_silence():
    assert rms_db(np.zeros(1000, dtype=np.float32)) == -120.0
    loud = np.ones(1000, dtype=np.float32) * 0.5
    assert rms_db(loud) > -10
