"""目标歌词对齐测试：lrc 解析、DTW 单调匹配与等长保证（不依赖真实模型）。"""

import numpy as np
import pytest

from core.errors import AlignmentFailed
from core.lyric_align import dtw_match, parse_lrc


def test_parse_lrc_skips_title_and_punct(tmp_path):
    lrc = tmp_path / "target.lrc"
    lrc.write_text(
        "[00:00.00]《干啥子嘛》（填词｜原曲：慌啥子嘛）\n"
        "[00:01.00]干啥子嘛 动起来\n"
        "[00:04.00]羽球飞起，烦恼都抛开！\n",
        encoding="utf-8",
    )
    parsed = parse_lrc(str(lrc))
    assert len(parsed["lines"]) == 2  # 标题行被剔除
    assert parsed["chars"] == "干啥子嘛动起来羽球飞起烦恼都抛开"
    assert " " not in parsed["chars"] and "，" not in parsed["chars"]


def test_parse_lrc_rejects_empty(tmp_path):
    lrc = tmp_path / "empty.lrc"
    lrc.write_text("没有时间戳的一行\n", encoding="utf-8")
    with pytest.raises(AlignmentFailed):
        parse_lrc(str(lrc))


def test_dtw_match_exact_one_to_one():
    mids = np.array([0.1, 0.2, 0.3, 0.4])
    matches, skipped = dtw_match(mids, mids)
    assert matches == [0, 1, 2, 3]
    assert skipped == []


def test_dtw_match_skips_duplicate_asr_char():
    # ASR 在 0.2s 处多识别出一个重复字，目标只有 4 字
    asr = np.array([0.1, 0.2, 0.21, 0.3, 0.4])
    target = np.array([0.1, 0.2, 0.3, 0.4])
    matches, skipped = dtw_match(asr, target)
    assert len(matches) == 4
    assert len(skipped) == 1
    assert skipped[0] in (1, 2)  # 重复位置二选一
    # 单调性严格递增
    assert all(matches[i] < matches[i + 1] for i in range(3))


def test_dtw_match_monotonic_on_jittered_sequence():
    rng = np.random.default_rng(7)
    base = np.arange(20, dtype=float) * 0.5
    target = base + rng.normal(0, 0.02, 20)
    asr_extra = np.sort(np.concatenate([base, [3.1, 7.7]]))  # 插入 2 个噪声字
    matches, skipped = dtw_match(asr_extra, target)
    assert len(matches) == 20
    assert len(skipped) == 2
    assert all(matches[i] < matches[i + 1] for i in range(19))


def test_dtw_match_fails_when_target_longer():
    with pytest.raises(AlignmentFailed):
        dtw_match(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3]))
