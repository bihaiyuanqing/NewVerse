"""音频读写与拼接。

约定：
- 所有数组 float32；单声道为一维 (samples,)，双声道为二维 (channels, samples)
- 统一采样率 44100，中间产物落盘 wav
- splice 两端各 50ms 线性交叉淡入淡出，硬约束，不允许改短（文档第六节（一））
"""

import os

import librosa
import numpy as np
import soundfile as sf

from core.errors import SpliceError
from core.schema import SR

# 交叉淡化时长硬约束（秒）
CROSSFADE_MS = 50
CROSSFADE_SEC = CROSSFADE_MS / 1000.0


def read_audio(path: str, sr: int = SR, mono: bool = False) -> np.ndarray:
    """读取音频并重采样到 sr。mono=True 返回一维，否则 (channels, samples)。"""

    audio, orig_sr = sf.read(path, always_2d=True, dtype="float32")
    # soundfile 返回 (samples, channels)，统一转为 (channels, samples)
    audio = audio.T
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    if mono:
        if audio.shape[0] > 1:
            audio = librosa.to_mono(audio)
        else:
            audio = audio[0]
        return audio.astype(np.float32)
    return audio.astype(np.float32)


def write_audio(path: str, audio: np.ndarray, sr: int = SR) -> str:
    """写出 wav。一维按单声道，二维按 (channels, samples) 转 soundfile 格式。"""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if audio.ndim == 1:
        out = audio
    else:
        out = audio.T
    sf.write(path, out, sr, subtype="PCM_16")
    return path


def cut(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """按秒切片，边界非法即报错。"""

    n = audio.shape[-1]
    a, b = round(start * sr), round(end * sr)
    if start < 0 or end <= start or b > n:
        raise SpliceError(
            f"切片越界：start={start:.3f}s end={end:.3f}s，音频总长 {n / sr:.3f}s"
        )
    return audio[..., a:b]


def splice(
    base: np.ndarray,
    replacement: np.ndarray,
    sr: int,
    start: float,
    end: float,
    fade_ms: int = CROSSFADE_MS,
) -> np.ndarray:
    """把 base 的 [start, end] 段替换为 replacement，两端线性交叉淡化。

    硬约束：fade_ms 不得小于 50ms；replacement 长度必须与被替换区间一致
    （时长对齐由 sing/mix 阶段负责），否则抛 SpliceError，不静默伸缩。
    """

    if fade_ms < CROSSFADE_MS:
        raise SpliceError(f"交叉淡化不得短于 {CROSSFADE_MS}ms，当前 {fade_ms}ms")

    n = base.shape[-1]
    a, b = round(start * sr), round(end * sr)
    if start < 0 or end <= start or a < 0 or b > n:
        raise SpliceError(
            f"替换边界越界：start={start:.3f}s end={end:.3f}s，音频总长 {n / sr:.3f}s"
        )

    region_len = b - a
    if replacement.shape[-1] != region_len:
        raise SpliceError(
            f"替换段长度不匹配：区间 {region_len} 采样，替换音频 {replacement.shape[-1]} 采样"
        )

    fade_n = min(int(fade_ms / 1000.0 * sr), region_len // 2)
    if fade_n <= 0:
        raise SpliceError("替换区间过短，无法完成交叉淡化")

    out = base.copy()
    fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    if base.ndim == 1:
        # 头部淡入：原声淡出 + 新声淡入
        out[a : a + fade_n] = base[a : a + fade_n] * fade_out + replacement[:fade_n] * fade_in
        # 中段整体替换
        out[a + fade_n : b - fade_n] = replacement[fade_n:-fade_n]
        # 尾部淡出：新声淡出 + 原声淡入
        out[b - fade_n : b] = replacement[-fade_n:] * fade_out + base[b - fade_n : b] * fade_in
    else:
        out[:, a : a + fade_n] = (
            base[:, a : a + fade_n] * fade_out + replacement[:, :fade_n] * fade_in
        )
        out[:, a + fade_n : b - fade_n] = replacement[:, fade_n:-fade_n]
        out[:, b - fade_n : b] = (
            replacement[:, -fade_n:] * fade_out + base[:, b - fade_n : b] * fade_in
        )
    return out


def rms_db(audio: np.ndarray) -> float:
    """计算 RMS 能量（dBFS），用于分离后的人声能量检查。"""

    x = audio.astype(np.float64)
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * np.log10(rms)
