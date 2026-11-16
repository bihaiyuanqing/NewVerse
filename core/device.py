"""设备自动探测与各推理组件的能力映射。

选择优先级：环境变量 NEWVERSE_DEVICE 强制指定 > CUDA > MPS > CPU。
不同组件后端支持不同，不能一刀切：
- Demucs / RMVPE(torch)：cuda / mps / cpu
- WhisperX（CTranslate2）：仅 cuda / cpu，Mac 上走 CPU
- YingMusic-Singer-Plus（CUDA 工具链，含 bitsandbytes）：仅 cuda / cpu
"""

import os
import torch


def select_device() -> torch.device:
    """通用 torch 设备：cuda > mps > cpu，支持 NEWVERSE_DEVICE 覆盖。"""

    forced = os.getenv("NEWVERSE_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_kind() -> str:
    return select_device().type


def is_cuda() -> bool:
    return torch.cuda.is_available() and os.getenv("NEWVERSE_DEVICE", "").strip().lower() != "cpu"


def demucs_device() -> torch.device:
    """Demucs 设备。MPS 上个别算子不支持时，由 separate.py 捕获后回退 CPU。"""

    return select_device()


def f0_device() -> torch.device:
    """音高提取（torchcrepe / RMVPE）设备。"""

    return select_device()


def whisper_device() -> str:
    """WhisperX / CTranslate2 只认 cuda 或 cpu，不支持 mps。"""

    return "cuda" if is_cuda() else "cpu"


def whisper_compute_type() -> str:
    """GPU 用 float16，CPU 用 int8（CTranslate2 官方推荐组合）。"""

    return "float16" if is_cuda() else "int8"


def sing_device() -> torch.device:
    """重唱模型仅支持 cuda / cpu；无 NVIDIA 卡时回退 CPU（速度不保证）。"""

    if is_cuda():
        return torch.device("cuda")
    return torch.device("cpu")


def describe() -> dict:
    """供日志与界面展示当前设备决策。"""

    return {
        "selected": device_kind(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "whisper_device": whisper_device(),
        "whisper_compute_type": whisper_compute_type(),
        "sing_device": sing_device().type,
        "forced_by_env": os.getenv("NEWVERSE_DEVICE", ""),
    }
