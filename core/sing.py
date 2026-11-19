"""单句重唱：封装 YingMusic-Singer-Plus。

架构决策：YingMusic 依赖树庞大且与 P0 环境存在版本冲突（numba/transformers 等），
故用独立 venv 以子进程方式调用 deploy/sing_adapter.py，本模块只负责：
- 切出原句人声片段作为 ref/melody（lyric edit 模式二者相同）
- 子进程超时强杀，异常统一转 SingFailed 中文
- 输出 44.1kHz 立体声 downmix 为单声道
- 时长对齐到原句采样数（保调变速），极端偏差直接报错而非静默处理

提供两种入口：
- sing_line：单句（独立子进程，含冷启动）
- sing_lines：批量（模型只加载一次，全曲重唱用，避免 N 次冷启动）
"""

import json
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
BATCH_TIMEOUT = 1800.0
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


def _check_env() -> None:
    if not os.path.exists(YING_PYTHON):
        raise SingFailed(f"YingMusic 解释器不存在：{YING_PYTHON}，请先完成 GPU 机 P1 环境部署")
    if not os.path.exists(os.path.join(YING_REPO, "src")):
        raise SingFailed(f"YingMusic 仓库不存在或不完整：{YING_REPO}")
    if not os.path.exists(SING_ADAPTER):
        raise SingFailed(f"重唱适配器不存在：{SING_ADAPTER}")


def _line_paths(work_dir: str, idx: int) -> dict:
    # 子进程 cwd 是 YingMusic 仓库根，这里必须用绝对路径，避免相对路径错位
    lines_dir = os.path.abspath(os.path.join(work_dir, "lines"))
    return {
        "dir": lines_dir,
        "ref": os.path.join(lines_dir, f"line_{idx:03d}_ref.wav"),
        "raw": os.path.join(lines_dir, f"line_{idx:03d}_raw.wav"),
        "new": os.path.join(lines_dir, f"line_{idx:03d}_new.wav"),
    }


def _post_process(raw_path: str, phrase, new_path: str) -> str:
    """raw 立体声 → 单声道 → 时长对齐原句 → 落盘，返回新声路径。"""

    if not os.path.exists(raw_path):
        raise SingFailed(f"第 {phrase.idx} 句重唱未产出音频文件：{raw_path}")
    raw = _downmix(read_audio(raw_path, sr=SR, mono=False))
    # 必须与 audio_io.splice 的区间口径一致：round(end*SR)-round(start*SR)，
    # 不能用 round(duration*SR)，浮点舍入可能差 1 采样导致 splice 报错
    target_samples = round(phrase.end * SR) - round(phrase.start * SR)
    aligned = fit_duration(raw, SR, target_samples)
    write_audio(new_path, aligned, SR)
    return new_path


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

    _check_env()
    paths = _line_paths(work_dir, phrase.idx)

    # 原句人声片段（单声道 44.1k），ref 与 melody 同一段
    vocal = read_audio(vocal_path, sr=SR, mono=True)
    phrase_clip = cut(vocal, SR, phrase.start, phrase.end)
    write_audio(paths["ref"], phrase_clip, SR)

    cmd = [
        YING_PYTHON, SING_ADAPTER,
        "--ref_audio", paths["ref"],
        "--ref_text", phrase.text,
        "--target_text", new_text,
        "--output", paths["raw"],
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

    new_path = _post_process(paths["raw"], phrase, paths["new"])
    return read_audio(new_path, sr=SR, mono=True), new_path


def sing_lines(
    doc,
    items: list,
    work_dir: str,
    *,
    timeout: float = BATCH_TIMEOUT,
    seed: int = 666,
) -> dict:
    """批量重唱：一次子进程加载模型，逐句推理。

    items: [(phrase, new_text), ...]
    返回 {phrase.idx: 新声 wav 路径}；任一句失败即抛 SingFailed 并附失败清单。
    """

    if not items:
        return {}
    _check_env()

    vocal = read_audio(doc.vocal_path, sr=SR, mono=True)
    tasks = []
    path_map = {}
    for phrase, new_text in items:
        paths = _line_paths(work_dir, phrase.idx)
        phrase_clip = cut(vocal, SR, phrase.start, phrase.end)
        write_audio(paths["ref"], phrase_clip, SR)
        path_map[phrase.idx] = paths
        tasks.append(
            {
                "idx": phrase.idx,
                "ref_audio": paths["ref"],
                "ref_text": phrase.text,
                "target_text": new_text,
                "output": paths["raw"],
                "seed": seed,
            }
        )

    lines_dir = os.path.abspath(os.path.join(work_dir, "lines"))
    tasks_path = os.path.join(lines_dir, "sing_tasks.json")
    results_path = os.path.join(lines_dir, "sing_results.json")
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    cmd = [
        YING_PYTHON, SING_ADAPTER,
        "--tasks", tasks_path,
        "--results", results_path,
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
        raise SingFailed(f"批量重唱超时（>{timeout:.0f}s），已终止子进程") from exc
    except OSError as exc:
        raise SingFailed(f"重唱子进程启动失败：{exc}") from exc

    if not os.path.exists(results_path):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise SingFailed(f"批量重唱未产出结果文件（退出码 {proc.returncode}）：" + " / ".join(tail))

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    failed = [r for r in results if not r["ok"]]
    if failed:
        detail = "；".join(f"第 {r['idx']} 句：{r['error']}" for r in failed)
        raise SingFailed(f"批量重唱 {len(failed)}/{len(results)} 句失败：{detail}")

    phrase_by_idx = {p.idx: p for p, _ in items}
    out_map = {}
    for r in results:
        phrase = phrase_by_idx[r["idx"]]
        paths = path_map[r["idx"]]
        out_map[r["idx"]] = _post_process(paths["raw"], phrase, paths["new"])
    return out_map
