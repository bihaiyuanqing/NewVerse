"""目标歌词（lrc）驱动的强制对齐。

解决的问题：自动 ASR 的断句/文本与用户给定的目标填词不一致，
直接按 ASR 句做同义词替换无法唱出目标歌词。本模块：

1. 解析 lrc，剔除标题行与标点，得到纯汉字目标文本（转简体）
2. 用 WhisperX 把目标文本强制对齐到人声轨，拿到每个目标字的时间戳
   （目标填词本就按原曲旋律等字数填写，时间槽位天然对应）
3. 沿用 analyze 的静音断句把目标字切成 Phrase
4. DTW 单调匹配：把每个目标字匹配到时间最近的原 ASR 字，
   ASR 多出的重复/误识别字显式跳过并报告，绝不删补目标字，
   保证每句 ref_text 与 target_text 严格等长，满足重唱模型约束
"""

import os
import re

import numpy as np
import zhconv

from core.analyze import (
    _build_char_units,
    _detect_silence_boundaries,
    _segment_phrases,
)
from core.errors import AlignmentFailed, call_with_timeout
from core.schema import CharUnit, Phrase, SongDoc, validate_song_doc

_LRC_LINE = re.compile(r"\[\d{2}:\d{2}(?:[.:]\d+)?\](.+)")
_HANZI = re.compile(r"[\u4e00-\u9fff]")
# 每跳过一个 ASR 字的 DTW 惩罚（秒）。正常相邻字时间距约 0.5~0.7s，
# 取 0.5s 使跳过只发生在重复字/误识别等时间近似的位置
DTW_SKIP_PENALTY = 0.5


def parse_lrc(lrc_path: str) -> dict:
    """解析 lrc，返回 {"lines": 每行原文, "chars": 纯汉字简体串}。

    标题行（含《》）与无时间戳行直接跳过；标点/空格不计入字数。
    """

    if not os.path.exists(lrc_path):
        raise AlignmentFailed(f"目标歌词文件不存在：{lrc_path}")

    lines = []
    with open(lrc_path, encoding="utf-8") as f:
        for raw in f:
            m = _LRC_LINE.match(raw.strip())
            if not m:
                continue
            content = m.group(1)
            if "《" in content or "》" in content:
                continue
            lines.append(content)

    if not lines:
        raise AlignmentFailed("lrc 未解析出任何带时间戳的歌词行")

    chars = zhconv.convert("".join(_HANZI.findall("".join(lines))), "zh-cn")
    if not chars:
        raise AlignmentFailed("lrc 解析后没有任何汉字")
    return {"lines": lines, "chars": chars}


def _align_target_chars(vocal_path: str, target_text: str, whisper_arch: str,
                        timeout: float) -> list:
    """WhisperX 强制对齐目标文本，返回原始 char 条目（含时间戳）。"""

    import soundfile as sf
    import whisperx

    from core.device import whisper_compute_type, whisper_device
    import core.analyze  # noqa: F401  触发 torch.load 兼容垫片

    device = whisper_device()
    info = sf.info(vocal_path)
    duration = info.frames / info.samplerate

    def _do():
        audio = whisperx.load_audio(vocal_path)
        align_model, metadata = whisperx.load_align_model(language_code="zh", device=device)
        return whisperx.align(
            [{"text": target_text, "start": 0.0, "end": duration}],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=True,
        )

    aligned = call_with_timeout(
        _do, timeout=timeout, timeout_msg=f"目标歌词对齐超时（>{timeout}s）"
    )
    chars = aligned["segments"][0].get("chars", []) or []
    raw = [
        {"ch": item["char"], "start": item.get("start"), "end": item.get("end")}
        for item in chars
        if item.get("start") is not None and item.get("end")
    ]
    if len(raw) != len(target_text):
        raise AlignmentFailed(
            f"目标歌词对齐字数异常：输入 {len(target_text)} 字，"
            f"拿到时间戳 {len(raw)} 字，请检查 lrc 与人声是否匹配"
        )
    return raw


def dtw_match(asr_mids: np.ndarray, target_mids: np.ndarray,
              skip_penalty: float = DTW_SKIP_PENALTY) -> tuple:
    """子序列 DTW：把 m 个目标字单调匹配到 n 个 ASR 字（n>=m）。

    返回 (matches, skipped)：
    - matches: 长度 m，matches[j] = 与第 j 个目标字匹配的 ASR 字下标
    - skipped: 未被匹配的 ASR 字下标列表（重复/误识别噪声）
    """

    n, m = len(asr_mids), len(target_mids)
    if n < m:
        raise AlignmentFailed(
            f"原 ASR 字数 {n} 少于目标歌词字数 {m}，无法在不补字的前提下对齐"
        )

    inf = float("inf")
    # dp[i][j]：前 i 个 ASR 字匹配前 j 个目标字、且第 j 个目标字匹配第 i 个 ASR 字的最小代价
    dp = np.full((n, m), inf, dtype=np.float64)
    # 首部跳过同样计惩罚，否则 DP 会倾向整体错位而跳过开头的真实字
    for i in range(n):
        dp[i, 0] = abs(asr_mids[i] - target_mids[0]) + i * skip_penalty

    for j in range(1, m):
        # best[i] = min over k<i of dp[k][j-1] + (i-1-k)*skip
        best = np.full(n, inf, dtype=np.float64)
        running = inf
        for i in range(n):
            if i >= 1:
                running = min(running + skip_penalty, dp[i - 1, j - 1])
            best[i] = running
        for i in range(j, n):
            dp[i, j] = abs(asr_mids[i] - target_mids[j]) + best[i]

    # 回溯（尾部跳过同样计惩罚）
    trailing = (n - 1 - np.arange(n)) * skip_penalty
    end = int(np.argmin(dp[:, m - 1] + trailing))
    matches = [0] * m
    matches[m - 1] = end
    j = m - 1
    i = end
    while j > 0:
        target_j = dp[i, j] - abs(asr_mids[i] - target_mids[j])
        prev_i = None
        best_val = inf
        for k in range(j - 1, i):
            val = dp[k, j - 1] + (i - 1 - k) * skip_penalty
            if abs(val - target_j) < 1e-9 and val < best_val:
                best_val, prev_i = val, k
        if prev_i is None:  # 数值兜底
            prev_i = int(np.argmin([
                dp[k, j - 1] + (i - 1 - k) * skip_penalty for k in range(j - 1, i)
            ])) + j - 1
        j -= 1
        i = prev_i
        matches[j] = i

    matched_set = set(matches)
    skipped = [i for i in range(n) if i not in matched_set]
    return matches, skipped


def apply_target_lyrics(doc: SongDoc, lrc_path: str, work_dir: str,
                        whisper_arch: str = "small", timeout: float = 900) -> tuple:
    """用 lrc 目标歌词重建 SongDoc，返回 (新 doc, 报告 dict)。

    新 doc 的每个 Phrase：
    - text：原 ASR 歌词（重唱 ref_text）
    - new_text：lrc 目标歌词（重唱 target_text），与 text 严格等长
    - chars：目标字的 CharUnit（时间戳来自强制对齐）
    """

    parsed = parse_lrc(lrc_path)
    target_text = parsed["chars"]

    raw_target = _align_target_chars(doc.vocal_path, target_text, whisper_arch, timeout)
    target_units = _build_char_units(raw_target)
    boundaries = _detect_silence_boundaries(doc.vocal_path)
    target_phrases = _segment_phrases(raw_target, target_units, boundaries)

    asr_units = [c for p in doc.phrases for c in p.chars]
    if len(asr_units) < len(target_units):
        raise AlignmentFailed(
            f"原 ASR 字数 {len(asr_units)} 少于目标歌词字数 {len(target_units)}，"
            "目标填词比原曲音节还多，无法等长重唱"
        )

    asr_mids = np.array([(u.start + u.end) / 2 for u in asr_units])
    tgt_mids = np.array([(u.start + u.end) / 2 for u in target_units])
    matches, skipped = dtw_match(asr_mids, tgt_mids)

    # 以目标字所属 Phrase 分组，同步取匹配到的原 ASR 字
    unit_to_phrase = {}
    for tp in target_phrases:
        for u in tp.chars:
            unit_to_phrase[id(u)] = tp.idx

    ref_by_phrase = {tp.idx: [] for tp in target_phrases}
    for t_idx, asr_idx in enumerate(matches):
        ref_by_phrase[unit_to_phrase[id(target_units[t_idx])]].append(asr_units[asr_idx].ch)

    new_phrases = []
    for tp in target_phrases:
        ref_text = "".join(ref_by_phrase[tp.idx])
        new_text = "".join(u.ch for u in tp.chars)
        if len(ref_text) != len(new_text):
            raise AlignmentFailed(
                f"第 {tp.idx} 句等长校验失败：原词 {len(ref_text)} 字 ≠ 目标 {len(new_text)} 字"
            )
        new_phrases.append(
            Phrase(
                idx=tp.idx,
                text=ref_text,
                start=tp.start,
                end=tp.end,
                chars=tp.chars,
                new_text=new_text,
                edited=False,
                new_vocal_path=None,
            )
        )

    new_doc = SongDoc(
        src_path=doc.src_path,
        vocal_path=doc.vocal_path,
        accomp_path=doc.accomp_path,
        sr=doc.sr,
        phrases=new_phrases,
    )
    validate_song_doc(new_doc)

    report = {
        "target_chars": len(target_units),
        "asr_chars": len(asr_units),
        "skipped_asr_idx": skipped,
        "skipped_asr_chars": [asr_units[i].ch for i in skipped],
        "phrases": len(new_phrases),
        "lrc_lines": len(parsed["lines"]),
    }
    os.makedirs(work_dir, exist_ok=True)
    new_doc.to_json(os.path.join(work_dir, "song_doc.json"))
    return new_doc, report
