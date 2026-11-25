"""分析 asr_verify_report：高可信问题句排序 + 异常句音频体检。"""
import json
import sys

import numpy as np
import soundfile as sf

WORK = sys.argv[1] if len(sys.argv) > 1 else "/root/NewVerse/output/gpu_run1"
r = json.load(open(f"{WORK}/asr_verify_report.json", encoding="utf-8"))
rows = r["rows"]

print("=== 基线>=75% 但重唱掉分的句子（高可信发音问题），按 Δ 升序 ===")
cand = [x for x in rows if x["ref_acc"] >= 0.75]
for x in sorted(cand, key=lambda x: x["new_acc"] - x["ref_acc"]):
    delta = x["new_acc"] - x["ref_acc"]
    print(f"  #{x['idx']:02d} 基线{x['ref_acc']:.0%}->新声{x['new_acc']:.0%} "
          f"Δ{delta:+.0%} 目标「{x['target']}」听「{x['asr_new']}」")

low = [x for x in rows if x["ref_acc"] <= 0.4]
print(f"\n=== 基线<=40%、ASR 无法判定的句数: {len(low)} -> {[x['idx'] for x in low]} ===")

print("\n=== 异常句音频体检（时长/RMS/峰值） ===")
for idx in (29, 27, 8, 15):
    for tag in ("ref", "new"):
        p = f"{WORK}/lines/line_{idx:03d}_{tag}.wav"
        d, sr = sf.read(p, dtype="float32")
        if d.ndim > 1:
            d = d.mean(1)
        rms = 20 * np.log10(np.sqrt(np.mean(d ** 2)) + 1e-12)
        pk = 20 * np.log10(np.max(np.abs(d)) + 1e-12)
        print(f"  #{idx:02d} {tag}: {len(d)/sr:.2f}s RMS={rms:.1f}dBFS peak={pk:.1f}dBFS")

print("\n=== 新声 vs 基线 准确率分布 ===")
import statistics
print("new_acc mean", round(statistics.mean(x["new_acc"] for x in rows), 3))
print("ref_acc mean", round(statistics.mean(x["ref_acc"] for x in rows), 3))
