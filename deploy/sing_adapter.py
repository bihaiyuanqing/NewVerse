"""YingMusic-Singer-Plus 单句重唱适配器。

在 YingMusic 独立 venv 内、以仓库根目录为 cwd 运行：
    python sing_adapter.py --ref_audio x.wav --ref_text 原句 \
        --target_text 新句 --output out.wav [--seed 666]

lyric edit 模式下 ref_audio 与 melody_audio 为同一段原句人声；
本脚本不调用 infer_api.download_files（无需 MelBandRoformer，分离已由 Demucs 完成），
直接走 PyTorchModelHubMixin.from_pretrained，离线环境依赖 HF 缓存。
"""

import argparse
import os
import sys

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


def main() -> int:
    parser = argparse.ArgumentParser(description="NewVerse 单句重唱适配器")
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--target_text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[sing_adapter] device={device} loading model ...", flush=True)
    model = YingMusicSinger.from_pretrained("ASLP-lab/YingMusic-Singer")
    model = model.to(device)
    model.eval()

    print(
        f"[sing_adapter] ref={args.ref_text!r} target={args.target_text!r}",
        flush=True,
    )
    with torch.inference_mode():
        audio, sr = model(
            ref_audio_path=args.ref_audio,
            melody_audio_path=args.ref_audio,
            ref_text=args.ref_text.strip(),
            target_text=args.target_text.strip(),
            lrc_align_mode="sentence_level",
            seed=args.seed,
        )

    torchaudio.save(args.output, audio.to("cpu"), sample_rate=sr)
    print(f"[sing_adapter] saved {args.output} sr={sr}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - 适配器边界，由父进程转中文异常
        print(f"[sing_adapter][error] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
