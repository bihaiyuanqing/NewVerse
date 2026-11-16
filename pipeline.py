"""流水线对外门面（文档第六节（七））。

界面层与 CLI 只允许调用本文件的三个函数，不得直接调用 core 内部函数。
P0 阶段实现 prepare；regenerate_line / render 在 P1 补齐。
"""

import os

from core.analyze import analyze, detect_harmony_risk
from core.separate import separate


def prepare(src_path: str, work_dir: str, **kwargs) -> dict:
    """分离 + 解析，返回 SongDoc 并落盘 song_doc.json。"""

    os.makedirs(work_dir, exist_ok=True)
    sep = separate(
        src_path,
        work_dir,
        model_name=kwargs.get("demucs_model", "htdemucs_ft"),
        timeout=kwargs.get("separate_timeout", 600),
    )
    doc = analyze(
        src_path=src_path,
        vocal_path=sep["vocal_path"],
        accomp_path=sep["accomp_path"],
        work_dir=work_dir,
        whisper_arch=kwargs.get("whisper_model", "small"),
        crepe_model=kwargs.get("crepe_model", "full"),
        timeout=kwargs.get("analyze_timeout", 900),
    )
    harmony = detect_harmony_risk([c for p in doc.phrases for c in p.chars])
    return {"doc": doc, "separation": sep, "harmony_warning": harmony}


def regenerate_line(doc, idx: int, theme: str, work_dir: str):
    """P1 实现：改一句，调 rewrite 再调 sing。"""
    raise NotImplementedError("P1 阶段实现")


def render(doc, out_path: str):
    """P1/P2 实现：合成整曲。"""
    raise NotImplementedError("P1 阶段实现")
