"""解析模块（文档第六节（三））。

三步：
1. WhisperX 中文转写 + 字级强制对齐
2. 断句：标点 + 静音间隔（>0.25s）双重切分，再按单句不超过 12 字硬切
3. 音高：torchcrepe 对整条人声轨一次性提 F0（10ms 帧移），再按字边界切片

说明：文档允许 RMVPE 或 torchcrepe 二选一。torchcrepe 是纯 pip 包、
跨平台（cuda/mps/cpu），本机先用它；GPU 机如需 RMVPE 可在本模块扩展，
schema 与下游不变。
"""

import math
import os

import numpy as np
import torch

from core.audio_io import read_audio
from core.device import f0_device, whisper_compute_type, whisper_device
from core.errors import AlignmentFailed, call_with_timeout
from core.schema import F0_HOP_MS, SR, CharUnit, Phrase, SongDoc, validate_song_doc

# 断句硬约束
SILENCE_GAP_SEC = 0.25
MAX_PHRASE_CHARS = 12
# F0 提取采样率与帧移（10ms）
F0_SR = 16000
F0_HOP = F0_SR * F0_HOP_MS // 1000  # 160 采样
# 相邻帧音高跳变超过该八度比例视为疑似双 F0（和声）候选，阈值待实测
HARMONY_OCTAVE_JUMP = 0.9
HARMONY_RISK_RATIO = 0.15

PUNCTUATION = set("，。！？、；：,.!?;:…—·~～\"'“”‘’()（）【】[]《》<>")


def _transcribe_chars(vocal_path: str, whisper_arch: str, timeout: float) -> list:
    """跑 WhisperX，返回带时间戳的原始 char 列表（含标点条目）。"""

    import whisperx

    device = whisper_device()
    compute_type = whisper_compute_type()

    def _do():
        model = whisperx.load_model(
            whisper_arch, device=device, compute_type=compute_type, language="zh"
        )
        audio = whisperx.load_audio(vocal_path)
        result = model.transcribe(audio, language="zh", batch_size=16)
        align_model, metadata = whisperx.load_align_model(
            language_code="zh", device=device
        )
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=True,
        )
        return aligned["segments"]

    segments = call_with_timeout(
        _do,
        timeout=timeout,
        timeout_msg=f"歌词对齐超时（>{timeout}s）",
    )

    raw_chars = []
    for seg in segments:
        for item in seg.get("chars", []) or []:
            ch = item.get("char")
            if ch is None:
                continue
            raw_chars.append(
                {
                    "ch": ch,
                    "start": item.get("start"),
                    "end": item.get("end"),
                }
            )
    if not raw_chars:
        raise AlignmentFailed("对齐结果为空，未能识别出任何人声歌词，请检查素材或更换对齐模型")
    return raw_chars


def _build_char_units(raw_chars: list) -> list:
    """过滤无时间戳条目，生成全局编号的 CharUnit（暂不含 f0）。"""

    units = []
    idx = 0
    for item in raw_chars:
        ch = item["ch"]
        start, end = item["start"], item["end"]
        if not ch or not ch.strip():
            continue
        if start is None or end is None or end <= start:
            # 标点等无时间戳条目只参与断句，不进 CharUnit
            continue
        units.append(CharUnit(idx=idx, ch=ch.strip(), start=float(start), end=float(end)))
        idx += 1
    if not units:
        raise AlignmentFailed("字级对齐失败：没有拿到带时间戳的汉字，建议更换对齐模型后重试")
    return units


def _segment_phrases(raw_chars: list, units: list) -> list:
    """标点 + 静音间隔双重切分，再按 12 字硬切。"""

    # 以带时间戳的字为骨架，标点信息从 raw_chars 的相邻关系里取
    timed = [u for u in units]
    phrases = []
    current = []
    prev_end = None
    unit_cursor = 0

    for item in raw_chars:
        ch, start, end = item["ch"], item["start"], item["end"]
        is_punct = ch in PUNCTUATION
        if is_punct:
            if current:
                phrases.append(current)
                current = []
            continue
        if start is None or end is None or end <= start:
            continue
        unit = timed[unit_cursor]
        unit_cursor += 1

        need_cut = False
        if prev_end is not None and unit.start - prev_end > SILENCE_GAP_SEC:
            need_cut = True
        if len(current) >= MAX_PHRASE_CHARS:
            need_cut = True
        if need_cut and current:
            phrases.append(current)
            current = []
        current.append(unit)
        prev_end = unit.end

    if current:
        phrases.append(current)

    result = []
    for i, group in enumerate(phrases):
        result.append(
            Phrase(
                idx=i,
                text="".join(c.ch for c in group),
                start=group[0].start,
                end=group[-1].end,
                chars=group,
            )
        )
    return result


def _extract_f0(vocal_path: str, crepe_model: str, timeout: float) -> np.ndarray:
    """整轨一次性提 F0，返回逐帧（10ms）基频，清音为 nan。"""

    import torchcrepe

    audio = read_audio(vocal_path, sr=F0_SR, mono=True)
    tensor = torch.from_numpy(audio).float().unsqueeze(0)
    device = f0_device()

    def _do():
        return torchcrepe.predict(
            tensor,
            F0_SR,
            hop_length=F0_HOP,
            model=crepe_model,
            device=device,
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=False,
        )

    try:
        f0 = call_with_timeout(
            _do, timeout=timeout, timeout_msg=f"音高提取超时（>{timeout}s）"
        )
    except Exception:
        if device.type == "mps":
            # MPS 不兼容时回退 CPU
            f0 = call_with_timeout(
                _do, timeout=timeout, timeout_msg=f"音高提取超时（>{timeout}s）"
            )
        else:
            raise

    return f0.squeeze(0).detach().cpu().numpy().astype("float64")


def _attach_f0(units: list, f0_frames: np.ndarray) -> None:
    """按字边界切 F0，nan 转 None，帧移固定 10ms。"""

    for unit in units:
        start_frame = math.floor(unit.start * 1000.0 / F0_HOP_MS)
        end_frame = math.ceil(unit.end * 1000.0 / F0_HOP_MS)
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, len(f0_frames))
        values = []
        for fi in range(start_frame, end_frame):
            value = f0_frames[fi]
            values.append(None if (value is None or math.isnan(value)) else float(value))
        unit.f0 = values


def detect_harmony_risk(units: list) -> dict:
    """和声风险粗检（文档第七节）：统计相邻帧八度级跳变占比，阈值待实测。"""

    voiced = []
    for unit in units:
        for value in unit.f0 or []:
            if value:
                voiced.append(value)
    if len(voiced) < 100:
        return {"risk": False, "jump_ratio": 0.0}

    jumps = 0
    for a, b in zip(voiced[:-1], voiced[1:]):
        if abs(math.log2(max(b, 1e-6) / max(a, 1e-6))) > HARMONY_OCTAVE_JUMP:
            jumps += 1
    ratio = jumps / (len(voiced) - 1)
    return {"risk": ratio > HARMONY_RISK_RATIO, "jump_ratio": ratio}


def analyze(
    src_path: str,
    vocal_path: str,
    accomp_path: str,
    work_dir: str,
    whisper_arch: str = "small",
    crepe_model: str = "full",
    timeout: float = 900,
) -> SongDoc:
    """解析人声轨，产出完整 SongDoc 并落盘 song_doc.json。"""

    if not os.path.exists(vocal_path):
        raise AlignmentFailed(f"人声文件不存在：{vocal_path}")

    raw_chars = _transcribe_chars(vocal_path, whisper_arch, timeout)
    units = _build_char_units(raw_chars)
    phrases = _segment_phrases(raw_chars, units)
    f0_frames = _extract_f0(vocal_path, crepe_model, timeout)
    _attach_f0(units, f0_frames)

    doc = SongDoc(
        src_path=src_path,
        vocal_path=vocal_path,
        accomp_path=accomp_path,
        sr=SR,
        phrases=phrases,
    )
    validate_song_doc(doc)

    os.makedirs(work_dir, exist_ok=True)
    doc.to_json(os.path.join(work_dir, "song_doc.json"))
    return doc
