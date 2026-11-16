"""混音：替换后的人声轨加轻混响，再与伴奏合成双声道。

- 只对人声做低湿声比卷积混响，让重唱句与原轨听感融合，不动伴奏
- 峰值超 1 时按峰值归一化，避免削波
"""

import os

import numpy as np
from scipy.signal import fftconvolve

from core.audio_io import read_audio, splice, write_audio
from core.errors import MixFailed
from core.schema import SR

REVERB_WET = 0.15        # 湿声比，轻混响
REVERB_IR_SEC = 0.8      # 脉冲响应时长
REVERB_DECAY = 4.0       # 指数衰减系数
VOCAL_GAIN = 1.0


def _build_impulse(sr: int, ir_sec: float, decay: float) -> np.ndarray:
    """生成指数衰减白噪声脉冲响应（确定性随机种子，保证可复现）。"""

    rng = np.random.default_rng(20261116)
    n = int(sr * ir_sec)
    t = np.linspace(0.0, ir_sec, n, dtype=np.float32)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-decay * t)
    norm = np.max(np.abs(ir))
    return ir / norm if norm > 0 else ir


def apply_light_reverb(
    audio: np.ndarray,
    sr: int = SR,
    wet: float = REVERB_WET,
    ir_sec: float = REVERB_IR_SEC,
    decay: float = REVERB_DECAY,
) -> np.ndarray:
    """单声道人声加轻混响，输出长度与输入一致。"""

    if audio.ndim != 1:
        raise MixFailed("混响输入应为单声道人声")
    ir = _build_impulse(sr, ir_sec, decay)
    wet_track = fftconvolve(audio, ir, mode="full")[: audio.shape[0]].astype(np.float32)
    mixed = (1.0 - wet) * audio + wet * wet_track
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:  # 混响自身也可能推高峰值，先压回安全区间
        mixed = mixed / peak
    return mixed.astype(np.float32)


def render_mix(doc, out_path: str, *, reverb: bool = True) -> str:
    """按 SongDoc 的编辑结果合成整曲，输出 44.1kHz 立体声 wav。"""

    if not doc.vocal_path or not doc.accomp_path:
        raise MixFailed("SongDoc 缺少人声/伴奏路径，无法混音")
    if not os.path.exists(doc.vocal_path) or not os.path.exists(doc.accomp_path):
        raise MixFailed("人声或伴奏文件不存在，请先重新 prepare")

    vocal = read_audio(doc.vocal_path, sr=SR, mono=True)
    edited = [p for p in doc.phrases if p.edited and p.new_vocal_path]
    for phrase in edited:
        if not os.path.exists(phrase.new_vocal_path):
            raise MixFailed(f"第 {phrase.idx} 句标记已编辑但新声文件缺失：{phrase.new_vocal_path}")
        new_vocal = read_audio(phrase.new_vocal_path, sr=SR, mono=True)
        vocal = splice(vocal, new_vocal, SR, phrase.start, phrase.end)

    if reverb:
        vocal = apply_light_reverb(vocal, SR)

    accomp = read_audio(doc.accomp_path, sr=SR, mono=False)
    if accomp.ndim == 1:
        accomp = np.stack([accomp, accomp], axis=0)

    n = min(vocal.shape[0], accomp.shape[1])
    vocal_stereo = np.broadcast_to(vocal[None, :n], (2, n)).astype(np.float32) * VOCAL_GAIN
    mixed = accomp[:, :n] + vocal_stereo

    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return write_audio(out_path, mixed.astype(np.float32), SR)
