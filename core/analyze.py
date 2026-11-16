"""解析模块（文档第六节（三））。

三步：
1. WhisperX 中文转写 + 字级强制对齐
2. 断句：标点 + 静音间隔（>0.25s）双重切分，再按单句不超过 12 字硬切
3. 音高：torchcrepe 对整条人声轨一次性提 F0（10ms 帧移），再按字边界切片

说明：文档允许 RMVPE 或 torchcrepe 二选一。torchcrepe 是纯 pip 包、
跨平台（cuda/mps/cpu），本机先用它；GPU 机如需 RMVPE 可在本模块扩展，
schema 与下游不变。
"""

import functools
import math
import os

import librosa
import numpy as np
import torch
import zhconv

from core.audio_io import read_audio
from core.device import f0_device, whisper_compute_type, whisper_device
from core.errors import AlignmentFailed, call_with_timeout
from core.schema import F0_HOP_MS, SR, CharUnit, Phrase, SongDoc, validate_song_doc


def _patch_torch_load_compat() -> None:
    """torch 2.6 起 torch.load 默认 weights_only=True，导致 WhisperX 依赖的
    Pyannote VAD 旧 checkpoint（含 omegaconf.ListConfig 等）无法加载。

    本项目所有 checkpoint 均来自 HuggingFace 官方仓库等可信来源，
    这里恢复 2.6 之前的默认行为，仅做进程级兼容，不改动模型文件。
    """

    if getattr(torch.load, "_newverse_patched", False):
        return
    orig_load = torch.load

    @functools.wraps(orig_load)
    def _load(*args, **kwargs):
        # lightning 会显式传 weights_only=None，setdefault 拦不住，
        # 必须把 None 也归一为 False（恢复 2.6 之前的行为）
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)

    _load._newverse_patched = True
    torch.load = _load


_patch_torch_load_compat()

# 断句硬约束
SILENCE_GAP_SEC = 0.25
MAX_PHRASE_CHARS = 12
# F0 提取采样率与帧移（10ms）
F0_SR = 16000
F0_HOP = F0_SR * F0_HOP_MS // 1000  # 160 采样
# 相邻帧音高跳变超过该八度比例视为疑似双 F0（和声）候选，阈值待实测
HARMONY_OCTAVE_JUMP = 0.9
HARMONY_RISK_RATIO = 0.15
# periodicity 低于该值判为清音（帧级 F0 置 None），阈值待实测回填
UNVOICED_PERIODICITY = 0.2

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
        # wav2vec2 中文对齐模型输出常含繁体，统一转简体
        ch_simp = zhconv.convert(ch.strip(), "zh-cn")
        units.append(CharUnit(idx=idx, ch=ch_simp, start=float(start), end=float(end)))
        idx += 1
    if not units:
        raise AlignmentFailed("字级对齐失败：没有拿到带时间戳的汉字，建议更换对齐模型后重试")

    # WhisperX 对齐缺陷：长间奏前最后一个字的 end 会被拉长到整个间奏
    # （实测出现单字 16.7s），用自适应上限钳制，正常拖腔（本曲最长 1.8s）不受影响
    durations = np.array([u.end - u.start for u in units])
    cap = max(CHAR_DURATION_CAP_SEC, float(np.median(durations)) * 3.0)
    for unit in units:
        if unit.end - unit.start > cap:
            unit.end = unit.start + cap
    return units


# 能量静音检测参数（实测回填，见 README 实测区）：
# 文档原 SILENCE_GAP_SEC=0.25 针对字时间戳间隙设计，但 WhisperX 字时间戳首尾
# 相接、间隙恒为 0，改用音频能量包络后，legato 演唱的句间气口普遍只有
# 0.15~0.25s，-35dB/0.25s 会漏掉大部分气口；实测 -28dB/0.15s 能切出完整
# 歌词句（如「黄沙慢慢漫来」「一切都是最好的安排」），故能量检测用独立参数
SILENCE_RMS_DB = -28.0
AUDIO_SILENCE_MIN_SEC = 0.15
# 小于 MIN_PHRASE_CHARS 的碎片向邻句合并
MIN_PHRASE_CHARS = 4
# 碎片合并允许跨越的最大时间间隙（秒），超过视为真实静音切口，禁止粘合
MERGE_MAX_GAP_SEC = 1.0
# 单字时长硬上限（秒），与 3 倍中位时长取 max，用于钳制间奏拉长伪时间戳
CHAR_DURATION_CAP_SEC = 2.0


def _detect_silence_boundaries(vocal_path: str) -> list:
    """从人声轨能量包络检测静音段中点，作为断句候选边界。

    WhisperX 字级时间戳首尾相接（实测字间隙恒为 0），文档要求的
    「静音间隔 >0.25s」无法从时间戳得到，改为直接分析音频能量。
    """

    audio = read_audio(vocal_path, sr=SR, mono=True)
    frame_len = int(0.02 * SR)
    hop = int(0.01 * SR)
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=0) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    silent = db < SILENCE_RMS_DB

    boundaries = []
    i, n = 0, len(silent)
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if (j - i) * 0.01 >= AUDIO_SILENCE_MIN_SEC:
            boundaries.append((i + j) / 2.0 * 0.01)
        i = j
    return boundaries


def _silence_cut_indices(units: list, boundaries: list) -> set:
    """把静音中点映射为字间断点，返回需要在其前面切分的 unit 下标集合。"""

    cut_before = set()
    for midpoint in boundaries:
        j = next((k for k, u in enumerate(units) if u.start > midpoint), None)
        if j:  # j=0（首字前的前奏静音）和 None（曲尾静音）不产生断点
            cut_before.add(j)
    return cut_before


def _segment_phrases(raw_chars: list, units: list, silence_boundaries: list) -> list:
    """静音边界 + 标点双重切分，再按单句不超过 12 字硬切，最后合并碎片。"""

    cut_before = _silence_cut_indices(units, silence_boundaries)
    phrases = []
    current = []
    unit_cursor = 0

    for item in raw_chars:
        ch, start, end = item["ch"], item["start"], item["end"]
        if ch in PUNCTUATION:
            if current:
                phrases.append(current)
                current = []
            continue
        if start is None or end is None or end <= start:
            continue
        unit = units[unit_cursor]
        global_idx = unit_cursor
        unit_cursor += 1

        gap = unit.start - current[-1].end if current else 0.0
        if current and (
            global_idx in cut_before
            or len(current) >= MAX_PHRASE_CHARS
            or gap > MERGE_MAX_GAP_SEC
        ):
            phrases.append(current)
            current = []
        current.append(unit)

    if current:
        phrases.append(current)

    # 碎片合并：过短句并入邻句，合并后不得超过 12 字
    # 但禁止跨真实静音合并：硬切相邻字间隙≈0，静音切口间隙明显更大
    merged = []
    for group in phrases:
        gap = group[0].start - merged[-1][-1].end if merged else 0.0
        if merged and (
            len(group) < MIN_PHRASE_CHARS or len(merged[-1]) < MIN_PHRASE_CHARS
        ) and gap <= MERGE_MAX_GAP_SEC and len(merged[-1]) + len(group) <= MAX_PHRASE_CHARS:
            merged[-1].extend(group)
        else:
            merged.append(group)

    return [
        Phrase(
            idx=i,
            text="".join(c.ch for c in group),
            start=group[0].start,
            end=group[-1].end,
            chars=group,
        )
        for i, group in enumerate(merged)
    ]


def _extract_f0(
    vocal_path: str,
    crepe_model: str,
    timeout: float,
    batch_size: int = 512,
) -> np.ndarray:
    """整轨一次性提 F0，返回逐帧（10ms）基频，清音为 nan。

    必须分块送批：3 分钟人声约 1.8 万帧，单批在 MPS 上会产生 18G 级
    卷积缓冲导致 Invalid buffer size，batch_size 控制峰值显存/内存。
    """

    import torchcrepe

    audio = read_audio(vocal_path, sr=F0_SR, mono=True)
    tensor = torch.from_numpy(audio).float().unsqueeze(0)

    def _do(device: torch.device):
        # 同时取 periodicity，低置信帧判清音置 nan，满足 schema「清音为 None」
        pitch, periodicity = torchcrepe.predict(
            tensor,
            F0_SR,
            hop_length=F0_HOP,
            model=crepe_model,
            device=device,
            batch_size=batch_size,
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=True,
        )
        periodicity = torchcrepe.filter.median(periodicity, 3)
        pitch = torchcrepe.filter.mean(pitch, 3)
        pitch[periodicity < UNVOICED_PERIODICITY] = float("nan")
        return pitch

    device = f0_device()
    try:
        f0 = call_with_timeout(
            _do, timeout=timeout, timeout_msg=f"音高提取超时（>{timeout}s）", device=device
        )
    except Exception:
        if device.type == "mps":
            # MPS 算子/内存不兼容时回退 CPU 重试
            f0 = call_with_timeout(
                _do,
                timeout=timeout,
                timeout_msg=f"音高提取超时（>{timeout}s）",
                device=torch.device("cpu"),
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
    silence_boundaries = _detect_silence_boundaries(vocal_path)
    phrases = _segment_phrases(raw_chars, units, silence_boundaries)
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
