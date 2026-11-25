"""ASR 发音核验：识别重唱人声并与目标歌词逐字对比。

对每句分别识别：
- line_xxx_new.wav（重唱新声）→ 与目标歌词 new_text 对比，量化合成发音准确度
- line_xxx_ref.wav（原人声声）→ 与原词 text 对比，作为 ASR 在歌声上的基线误差，
  用于区分「合成发音问题」与「ASR 自身听错」。

输出 JSON 报告到 work_dir/asr_verify_report.json，并打印人读摘要。
在 GPU 机用 /opt/newverse-venv（含 faster_whisper/zhconv）运行。
"""

import difflib
import json
import os
import sys

from faster_whisper import WhisperModel
from zhconv import convert as to_simplified

WORK_DIR = sys.argv[1] if len(sys.argv) > 1 else "output/gpu_run1"
MODEL_SIZE = sys.argv[2] if len(sys.argv) > 2 else "small"

# 只保留 CJK 统一表意文字，去掉标点/空白/英文噪声后再比字
def cjk_chars(text: str) -> list:
    text = to_simplified(text or "", "zh-cn")
    return [c for c in text if "一" <= c <= "鿿"]


def transcribe(model, wav_path: str) -> str:
    segments, _ = model.transcribe(
        wav_path,
        language="zh",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return "".join(seg.text for seg in segments)


def align_diff(target: list, heard: list) -> dict:
    """逐字对齐，返回正确数与逐位置差异（目标字→听到字）。"""
    sm = difflib.SequenceMatcher(a=target, b=heard, autojunk=False)
    equal, mismatches = 0, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            equal += i2 - i1
        else:
            t = "".join(target[i1:i2]) or "∅"
            h = "".join(heard[j1:j2]) or "∅"
            mismatches.append({"op": tag, "target": t, "heard": h})
    return {"equal": equal, "mismatches": mismatches}


def main() -> int:
    doc = json.load(open(os.path.join(WORK_DIR, "song_doc.json"), encoding="utf-8"))
    phrases = doc["phrases"]
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    rows, new_total, new_correct, ref_total, ref_correct = [], 0, 0, 0, 0
    for p in phrases:
        idx = p["idx"]
        new_wav = os.path.join(WORK_DIR, "lines", f"line_{idx:03d}_new.wav")
        ref_wav = os.path.join(WORK_DIR, "lines", f"line_{idx:03d}_ref.wav")

        target = cjk_chars(p.get("new_text", ""))
        original = cjk_chars(p.get("text", ""))
        heard_new = cjk_chars(transcribe(model, new_wav)) if os.path.exists(new_wav) else []
        heard_ref = cjk_chars(transcribe(model, ref_wav)) if os.path.exists(ref_wav) else []

        new_cmp = align_diff(target, heard_new)
        ref_cmp = align_diff(original, heard_ref)

        new_total += len(target)
        new_correct += new_cmp["equal"]
        ref_total += len(original)
        ref_correct += ref_cmp["equal"]
        rows.append({
            "idx": idx,
            "target": "".join(target),
            "asr_new": "".join(heard_new),
            "new_acc": round(new_cmp["equal"] / max(len(target), 1), 3),
            "new_mismatch": new_cmp["mismatches"],
            "original": "".join(original),
            "asr_ref": "".join(heard_ref),
            "ref_acc": round(ref_cmp["equal"] / max(len(original), 1), 3),
        })

    report = {
        "model": MODEL_SIZE,
        "lines": len(rows),
        "new_vocal_char_acc": round(new_correct / max(new_total, 1), 4),
        "ref_vocal_char_acc_baseline": round(ref_correct / max(ref_total, 1), 4),
        "rows": rows,
    }
    out = os.path.join(WORK_DIR, "asr_verify_report.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n===== ASR 核验汇总（{MODEL_SIZE}） =====")
    print(f"重唱新声 字准确率: {report['new_vocal_char_acc']:.2%}（{new_correct}/{new_total}）")
    print(f"原人声声 基线准确率: {report['ref_vocal_char_acc_baseline']:.2%}（{ref_correct}/{ref_total}）")
    print("\n准确率低于 0.8 的句子（重点排查发音）：")
    for r in rows:
        if r["new_acc"] < 0.8:
            print(f"  #{r['idx']:02d} acc={r['new_acc']:.0%} 目标「{r['target']}」→ 听成「{r['asr_new']}」"
                  f"（原声基线 {r['ref_acc']:.0%}）")
    print(f"\n报告已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
