"""流水线对外门面（文档第六节（七））。

界面层与 CLI 只允许调用本文件的三个函数，不得直接调用 core 内部函数。
P0 阶段实现 prepare；regenerate_line / render 在 P1 补齐。
"""

import os
import time

import torch

from core.analyze import analyze, detect_harmony_risk
from core.device import device_kind
from core.separate import separate


def _peak_memory_mb() -> float:
    """当前设备峰值显存/内存（MB），仅 MPS/CUDA 可统计，CPU 返回 0。"""

    kind = device_kind()
    try:
        if kind == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        if kind == "mps":
            return torch.mps.current_allocated_memory() / 1024 / 1024
    except Exception:  # noqa: BLE001 - 统计失败不影响主流程
        pass
    return 0.0


def prepare(src_path: str, work_dir: str, **kwargs) -> dict:
    """分离 + 解析，返回 SongDoc 并落盘 song_doc.json。"""

    os.makedirs(work_dir, exist_ok=True)
    metrics = {"device": device_kind()}

    t0 = time.time()
    sep = separate(
        src_path,
        work_dir,
        model_name=kwargs.get("demucs_model", "htdemucs_ft"),
        timeout=kwargs.get("separate_timeout", 600),
    )
    metrics["separate_sec"] = round(time.time() - t0, 1)

    t0 = time.time()
    doc = analyze(
        src_path=src_path,
        vocal_path=sep["vocal_path"],
        accomp_path=sep["accomp_path"],
        work_dir=work_dir,
        whisper_arch=kwargs.get("whisper_model", "small"),
        crepe_model=kwargs.get("crepe_model", "full"),
        timeout=kwargs.get("analyze_timeout", 900),
    )
    metrics["analyze_sec"] = round(time.time() - t0, 1)
    metrics["peak_memory_mb"] = round(_peak_memory_mb(), 1)

    harmony = detect_harmony_risk([c for p in doc.phrases for c in p.chars])
    return {"doc": doc, "separation": sep, "harmony_warning": harmony, "metrics": metrics}


def regenerate_line(doc, idx: int, theme: str, work_dir: str):
    """P1 实现：改一句，调 rewrite 再调 sing。"""
    raise NotImplementedError("P1 阶段实现")


def render(doc, out_path: str):
    """P1/P2 实现：合成整曲。"""
    raise NotImplementedError("P1 阶段实现")
