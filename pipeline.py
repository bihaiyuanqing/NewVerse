"""流水线对外门面（文档第六节（七））。

界面层与 CLI 只允许调用本文件的三个函数，不得直接调用 core 内部函数。
P0 阶段实现 prepare；regenerate_line / render 在 P1 补齐。
"""

import os
import time

import torch

from core.analyze import analyze, detect_harmony_risk
from core.device import device_kind
from core.errors import NewVerseError
from core.lyric_align import apply_target_lyrics
from core.mix import render_mix
from core.rewrite import rewrite_line
from core.separate import separate
from core.sing import sing_line, sing_lines


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


def regenerate_line(doc, idx: int, theme: str, work_dir: str, **kwargs) -> dict:
    """改一句：等字数改词 → 单句重唱 → 回写 SongDoc 并落盘。

    界面/CLI 只允许通过本函数触发单句重生成，不得直接调 core。
    """

    if not 0 <= idx < len(doc.phrases):
        raise NewVerseError(f"句序号越界：{idx}，当前共 {len(doc.phrases)} 句")
    phrase = doc.phrases[idx]

    new_text = rewrite_line(
        phrase.text,
        phrase.n_syl,
        theme,
        mock=kwargs.get("mock", False),
        max_retry=kwargs.get("rewrite_retry", 3),
        timeout=kwargs.get("rewrite_timeout", 30.0),
    )

    t0 = time.time()
    _, new_vocal_path = sing_line(
        phrase,
        doc.vocal_path,
        new_text,
        work_dir,
        timeout=kwargs.get("sing_timeout", 300.0),
        seed=kwargs.get("seed", 666),
    )
    sing_sec = round(time.time() - t0, 1)

    phrase.new_text = new_text
    phrase.edited = True
    phrase.new_vocal_path = new_vocal_path
    doc.to_json(os.path.join(work_dir, "song_doc.json"))

    return {
        "idx": idx,
        "old_text": phrase.text,
        "new_text": new_text,
        "n_syl": phrase.n_syl,
        "new_vocal_path": new_vocal_path,
        "sing_sec": sing_sec,
    }


def regenerate_all(doc, theme: str, work_dir: str, **kwargs) -> dict:
    """全曲逐句改词重唱：每句先等字数改词，再一次性批量重唱（模型只加载一次）。

    返回汇总信息；改词后与原句相同的句子也会重唱（模型重唱本身即产生新声）。
    """

    items = []
    rewritten = []
    for phrase in doc.phrases:
        new_text = rewrite_line(
            phrase.text,
            phrase.n_syl,
            theme,
            mock=kwargs.get("mock", False),
            max_retry=kwargs.get("rewrite_retry", 3),
            timeout=kwargs.get("rewrite_timeout", 30.0),
        )
        items.append((phrase, new_text))
        rewritten.append({"idx": phrase.idx, "old": phrase.text, "new": new_text})

    t0 = time.time()
    out_map = sing_lines(
        doc,
        items,
        work_dir,
        timeout=kwargs.get("sing_timeout", 1800.0),
        seed=kwargs.get("seed", 666),
    )
    sing_sec = round(time.time() - t0, 1)

    for phrase, new_text in items:
        phrase.new_text = new_text
        phrase.edited = True
        phrase.new_vocal_path = out_map[phrase.idx]
    doc.to_json(os.path.join(work_dir, "song_doc.json"))

    changed = sum(1 for r in rewritten if r["old"] != r["new"])
    return {
        "total": len(items),
        "text_changed": changed,
        "sing_sec": sing_sec,
        "details": rewritten,
    }


def apply_lyrics(doc, lrc_path: str, work_dir: str, **kwargs) -> tuple:
    """用 lrc 目标歌词重建 SongDoc（强制对齐 + DTW 等长槽位）。"""

    return apply_target_lyrics(
        doc,
        lrc_path,
        work_dir,
        whisper_arch=kwargs.get("whisper_model", "small"),
        timeout=kwargs.get("align_timeout", 900.0),
    )


def resing_all(doc, work_dir: str, **kwargs) -> dict:
    """按 SongDoc 中已有的 new_text 全曲批量重唱（不做改词，词来自 apply_lyrics）。"""

    missing = [p.idx for p in doc.phrases if not p.new_text]
    if missing:
        raise NewVerseError(f"以下句子缺少目标歌词 new_text，无法重唱：{missing}")

    items = [(p, p.new_text) for p in doc.phrases]
    t0 = time.time()
    out_map = sing_lines(
        doc,
        items,
        work_dir,
        timeout=kwargs.get("sing_timeout", 1800.0),
        seed=kwargs.get("seed", 666),
    )
    sing_sec = round(time.time() - t0, 1)

    for phrase, _ in items:
        phrase.edited = True
        phrase.new_vocal_path = out_map[phrase.idx]
    doc.to_json(os.path.join(work_dir, "song_doc.json"))

    return {"total": len(items), "sing_sec": sing_sec}


def render(doc, out_path: str, **kwargs) -> str:
    """把全部已编辑句子替换回人声轨，加轻混响后与伴奏合成整曲。"""

    return render_mix(doc, out_path, reverb=kwargs.get("reverb", True))
