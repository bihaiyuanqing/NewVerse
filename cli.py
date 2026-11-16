"""命令行入口。P0：prepare 跑通分离与解析，并打印每句原文与字数（P0.3）。"""

import argparse
import sys

from core.device import describe
from core.errors import NewVerseError
from pipeline import prepare


def cmd_prepare(args) -> int:
    print(f"[设备] {describe()}")
    result = prepare(args.audio, args.work_dir)
    doc = result["doc"]

    print(f"\n分离产物：{result['separation']['vocal_path']}")
    print(f"伴奏产物：{result['separation']['accomp_path']}")
    print(f"人声 RMS：{result['separation']['rms_db']:.1f} dB")
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

    args = parser.parse_args()
    try:
        return args.func(args)
    except NewVerseError as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
