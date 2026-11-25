"""YingMusic-Singer-Plus 重唱适配器（支持单句与批量）。

单句模式（兼容旧调用）：
    python sing_adapter.py --ref_audio x.wav --ref_text 原句 \
        --target_text 新句 --output out.wav [--seed 666]

批量模式（全曲重唱，模型只加载一次）：
    python sing_adapter.py --tasks tasks.json --results results.json [--seed 666]
    tasks.json: [{"ref_audio","ref_text","target_text","output","idx"}, ...]
    results.json: [{"idx","ok","output"|"error","sec"}, ...]

lyric edit 模式下 ref_audio 与 melody_audio 为同一段原句人声；
本脚本不调用 infer_api.download_files（无需 MelBandRoformer，分离已由 Demucs 完成），
直接走 PyTorchModelHubMixin.from_pretrained，离线环境依赖 HF 缓存。
"""

import argparse
import json
import os
import sys
import time

# 脚本运行时 sys.path[0] 是本文件目录，需手动加入 cwd（YingMusic 仓库根）才能 import src
sys.path.insert(0, os.getcwd())

import phonemizer.backend as _phonemizer_backend

# 离线 GPU 机无 espeak-ng 系统库，内网 apt 源也不含该包。
# 中文 g2p 走 pypinyin + 仓库内置 BERT ONNX（mandarin.chinese_to_ipa 不调 espeak），
# 仅 PhonemeBpeTokenizer 初始化时会为各语种急切构造 EspeakBackend，这里用惰性桩替代：
# 构造不再失败；只有误走非中文音素化时才显式报错。
if not _phonemizer_backend.EspeakBackend.is_available():
    class _EspeakStub:
        def __init__(self, *args, **kwargs):
            self.language = kwargs.get("language", args[0] if args else "unknown")

        def phonemize(self, *args, **kwargs):
            raise RuntimeError(
                f"当前环境未安装 espeak-ng，语种 {self.language} 的音素化不可用；"
                "NewVerse 演示仅支持中文歌词"
            )

    _phonemizer_backend.EspeakBackend = _EspeakStub

import torch
import torchaudio


def _load_model(device: str):
    from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger

    # 模型可由环境变量覆盖，默认 Plus（与 Mac Gradio 联调保持一致）；
    # 离线 GPU 机依赖 HF 缓存，需提前把对应模型同步到 ~/.cache/huggingface/hub。
    model_id = os.environ.get("NEWVERSE_YING_MODEL", "ASLP-lab/YingMusic-Singer-Plus")
    print(f"[sing_adapter] device={device} model={model_id} loading ...", flush=True)
    model = YingMusicSinger.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model


def _synthesize(model, device: str, task: dict, seed: int) -> tuple:
    print(
        f"[sing_adapter] idx={task.get('idx')} ref={task['ref_text']!r} "
        f"target={task['target_text']!r}",
        flush=True,
    )
    with torch.inference_mode():
        audio, sr = model(
            ref_audio_path=task["ref_audio"],
            melody_audio_path=task["ref_audio"],
            ref_text=task["ref_text"].strip(),
            target_text=task["target_text"].strip(),
            lrc_align_mode="sentence_level",
            seed=task.get("seed", seed),
        )
    torchaudio.save(task["output"], audio.to("cpu"), sample_rate=sr)
    return sr


def main() -> int:
    parser = argparse.ArgumentParser(description="NewVerse 重唱适配器")
    parser.add_argument("--ref_audio")
    parser.add_argument("--ref_text")
    parser.add_argument("--target_text")
    parser.add_argument("--output")
    parser.add_argument("--tasks", help="批量任务 JSON 路径")
    parser.add_argument("--results", help="批量结果 JSON 路径")
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = _load_model(device)

    # —— 单句模式 ——
    if not args.tasks:
        if not all([args.ref_audio, args.ref_text, args.target_text, args.output]):
            print("[sing_adapter][error] 单句模式参数不完整", file=sys.stderr)
            return 2
        _synthesize(
            model,
            device,
            {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "target_text": args.target_text,
                "output": args.output,
            },
            args.seed,
        )
        print(f"[sing_adapter] saved {args.output}", flush=True)
        return 0

    # —— 批量模式：单句失败不拖垮整批 ——
    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    results = []
    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        try:
            sr = _synthesize(model, device, task, args.seed)
            results.append(
                {
                    "idx": task.get("idx"),
                    "ok": True,
                    "output": task["output"],
                    "sr": sr,
                    "sec": round(time.time() - t0, 1),
                }
            )
            print(f"[sing_adapter] {i}/{len(tasks)} done in {results[-1]['sec']}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - 单句隔离，错误交父进程转中文
            results.append(
                {
                    "idx": task.get("idx"),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "sec": round(time.time() - t0, 1),
                }
            )
            print(f"[sing_adapter] {i}/{len(tasks)} FAILED: {exc}", file=sys.stderr, flush=True)

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    failed = [r for r in results if not r["ok"]]
    if len(failed) == len(results):
        return 2
    return 3 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - 适配器边界，由父进程转中文异常
        print(f"[sing_adapter][error] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
