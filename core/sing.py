"""单句重唱：封装 YingMusic-Singer-Plus。

架构决策：YingMusic 依赖树庞大且与 P0 环境存在版本冲突（numba/transformers 等），
故用独立 venv 以子进程方式调用 deploy/sing_adapter.py，本模块只负责：
- 切出原句人声片段作为 ref/melody（lyric edit 模式二者相同）
- 子进程超时强杀，异常统一转 SingFailed 中文
- 输出 44.1kHz 立体声 downmix 为单声道
- 时长对齐到原句采样数（保调变速），极端偏差直接报错而非静默处理
"""

import os
import subprocess

import librosa
import numpy as np

from core.audio_io import cut, read_audio, write_audio
from core.errors import SingFailed
from core.schema import SR

# 环境变量覆盖默认部署路径
YING_REPO = os.environ.get("NEWVERSE_YING_REPO", "/root/YingMusic-Singer-Plus")
YING_PYTHON = os.environ.get("NEWVERSE_YING_PYTHON", "/opt/yingmusic-venv/bin/python")
SING_ADAPTER = os.environ.get(
    "NEWVERSE_SING_ADAPTER",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "sing_adapter.py"),
)
DEFAULT_TIMEOUT = 300.0
# 变速率保护：超过 2 倍伸缩说明合成严重异常，不再硬拉
MAX_STRETCH_RATE = 2.0


def fit_duration(audio: np.ndarray, sr: int, target_samples: int) -> np.ndarray:
    """把重唱音频对齐到目标采样数：保调变速 + 末端零填充/截断兜底。"""

    if audio.ndim != 1:
        raise SingFailed("重唱音频应为单声道，实际维度异常")
    current = audio.shape[0]
    if current == target_samples:
        return audio.astype(np.float32)

    rate = current / target_samples
    if rate > MAX_STRETCH_RATE or rate < 1.0 / MAX_STRETCH_RATE:
        raise SingFailed(
            f"重唱时长偏差过大（伸缩率 {rate:.2f}，超出 ±{MAX_STRETCH_RATE} 倍保护），"
            "请检查原句音频或改词文本"
        )

    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
    if stretched.shape[0] >= target_samples:
        return stretched[:target_samples].astype(np.float32)
    pad = np.zeros(target_samples - stretched.shape[0], dtype=np.float32)
    return np.concatenate([stretched, pad]).astype(np.float32)


def _downmix(audio: np.ndarray) -> np.ndarray:
    """(channels, samples) 立体声 downmix 为单声道。"""

    if audio.ndim == 1:
        return audio
    return librosa.to_mono(audio).astype(np.float32)


def sing_line(
    phrase,
    vocal_path: str,
    new_text: str,
    work_dir: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    seed: int = 666,
) -> tuple:
    """重唱一句，返回 (对齐后的单声道音频, 新声 wav 路径)。"""

    if not os.path.exists(YING_PYTHON):
        raise SingFailed(f"YingMusic 解释器不存在：{YING_PYTHON}，请先完成 GPU 机 P1 环境部署")
    if not os.path.exists(os.path.join(YING_REPO, "src")):
        raise SingFailed(f"YingMusic 仓库不存在或不完整：{YING_REPO}")

    # 子进程 cwd 是 YingMusic 仓库根，这里必须用绝对路径，避免相对路径错位
    lines_dir = os.path.abspath(os.path.join(work_dir, "lines"))
    ref_path = os.path.join(lines_dir, f"line_{phrase.idx:03d}_ref.wav")
    raw_path = os.path.join(lines_dir, f"line_{phrase.idx:03d}_raw.wav")
    new_path = os.path.join(lines_dir, f"line_{phrase.idx:03d}_new.wav")

    # 原句人声片段（单声道 44.1k），ref 与 melody 同一段
    vocal = read_audio(vocal_path, sr=SR, mono=True)
    phrase_clip = cut(vocal, SR, phrase.start, phrase.end)
    write_audio(ref_path, phrase_clip, SR)

    cmd = [
        YING_PYTHON, SING_ADAPTER,
        "--ref_audio", ref_path,
        "--ref_text", phrase.text,
        "--target_text", new_text,
        "--output", raw_path,
        "--seed", str(seed),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=YING_REPO,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise SingFailed(f"第 {phrase.idx} 句重唱超时（>{timeout:.0f}s），已终止子进程") from exc
    except OSError as exc:
        raise SingFailed(f"重唱子进程启动失败：{exc}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise SingFailed(
            f"第 {phrase.idx} 句重唱失败（退出码 {proc.returncode}）：" + " / ".join(tail)
        )
    if not os.path.exists(raw_path):
        raise SingFailed(f"第 {phrase.idx} 句重唱未产出音频文件：{raw_path}")

    raw = _downmix(read_audio(raw_path, sr=SR, mono=False))
    target_samples = round(phrase.duration * SR)
    aligned = fit_duration(raw, SR, target_samples)
    write_audio(new_path, aligned, SR)
    return aligned, new_path
