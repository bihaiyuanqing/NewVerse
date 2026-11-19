"""命令行入口。P0：prepare 跑通分离与解析，并打印每句原文与字数（P0.3）。"""

import argparse
import sys

from core.device import describe
from core.errors import NewVerseError
from core.schema import SongDoc
from pipeline import prepare, regenerate_all, regenerate_line, render


def cmd_prepare(args) -> int:
    print(f"[设备] {describe()}")
    result = prepare(args.audio, args.work_dir)
    doc = result["doc"]

    metrics = result.get("metrics", {})
    print(f"\n分离产物：{result['separation']['vocal_path']}")
    print(f"伴奏产物：{result['separation']['accomp_path']}")
    print(f"人声 RMS：{result['separation']['rms_db']:.1f} dB")
    print(
        f"实测耗时：分离 {metrics.get('separate_sec')}s + 解析 {metrics.get('analyze_sec')}s"
        f"（设备 {metrics.get('device')}，峰值显存 {metrics.get('peak_memory_mb')}MB）"
    )
    if result["harmony_warning"]["risk"]:
        print(
            "[警告] 检测到疑似双 F0（和声）风险，跳变占比 "
            f"{result['harmony_warning']['jump_ratio']:.2%}，建议更换素材"
        )

    print(f"\n共 {len(doc.phrases)} 句：")
    for phrase in doc.phrases:
        print(f"  第 {phrase.idx:>2} 句 | {phrase.n_syl:>2} 字 | {phrase.text}")
    print(f"\nSongDoc 已落盘：{args.work_dir}/song_doc.json")
    return 0


def cmd_regenerate(args) -> int:
    doc = SongDoc.from_json(f"{args.work_dir}/song_doc.json")
    result = regenerate_line(
        doc, args.idx, args.theme, args.work_dir, mock=args.mock, seed=args.seed
    )
    print(
        f"第 {result['idx']} 句重唱完成（{result['n_syl']} 字，耗时 {result['sing_sec']}s）\n"
        f"  原句：{result['old_text']}\n"
        f"  新词：{result['new_text']}\n"
        f"  新声：{result['new_vocal_path']}"
    )
    return 0


def cmd_regenerate_all(args) -> int:
    doc = SongDoc.from_json(f"{args.work_dir}/song_doc.json")
    result = regenerate_all(doc, args.theme, args.work_dir, mock=args.mock, seed=args.seed)
    print(f"全曲重唱完成：共 {result['total']} 句，其中 {result['text_changed']} 句文本发生变化，"
          f"重唱总耗时 {result['sing_sec']}s")
    for item in result["details"]:
        flag = " " if item["old"] == item["new"] else "✎"
        print(f"  {flag} #{item['idx']:02d} {item['old']} → {item['new']}")
    return 0


def cmd_render(args) -> int:
    doc = SongDoc.from_json(f"{args.work_dir}/song_doc.json")
    edited = sum(1 for p in doc.phrases if p.edited)
    if edited == 0:
        print("[警告] 当前没有任何已编辑句子，输出将与原曲一致（仍走轻混响）")
    out_path = render(doc, args.output, reverb=not args.no_reverb)
    print(f"混音完成：共替换 {edited} 句，输出 {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NewVerse 音乐改词 Demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="分离 + 解析，产出 SongDoc")
    p_prepare.add_argument("audio", help="输入音频路径（mp3/wav/m4a）")
    p_prepare.add_argument("-o", "--work-dir", default="output/run", help="工作目录")
    p_prepare.add_argument("--whisper-model", default="small")
    p_prepare.add_argument("--crepe-model", default="full")
    p_prepare.add_argument("--demucs-model", default="htdemucs_ft")
    p_prepare.set_defaults(func=cmd_prepare)

    p_regen = sub.add_parser("regenerate", help="改写并重唱指定句")
    p_regen.add_argument("idx", type=int, help="句序号（从 0 开始）")
    p_regen.add_argument("-o", "--work-dir", default="output/run")
    p_regen.add_argument("-t", "--theme", default="", help="改词主题")
    p_regen.add_argument("--mock", action="store_true", help="强制使用内置 Mock 改词")
    p_regen.add_argument("--seed", type=int, default=666)
    p_regen.set_defaults(func=cmd_regenerate)

    p_all = sub.add_parser("regenerate-all", help="全曲逐句改词并重唱（模型只加载一次）")
    p_all.add_argument("-o", "--work-dir", default="output/run")
    p_all.add_argument("-t", "--theme", default="", help="改词主题")
    p_all.add_argument("--mock", action="store_true", help="强制使用内置 Mock 改词")
    p_all.add_argument("--seed", type=int, default=666)
    p_all.set_defaults(func=cmd_regenerate_all)

    p_render = sub.add_parser("render", help="合成整曲")
    p_render.add_argument("-o", "--work-dir", default="output/run")
    p_render.add_argument("output", help="输出 wav 路径")
    p_render.add_argument("--no-reverb", action="store_true", help="关闭轻混响")
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    try:
        return args.func(args)
    except NewVerseError as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
