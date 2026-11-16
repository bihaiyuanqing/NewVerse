#!/usr/bin/env bash
# 第一步（在有外网的 Mac 上执行）：准备 GPU 机无法从外网获取的物料并传输。
# 实测背景：GPU 机仅通内网镜像（bytedpypi / mirrors.byted.org/nvidia），
# 不通 GitHub / HuggingFace / python.org / johnvansickle。
# 用法：GPU_HOST=n194-194-195 bash deploy/01_prepare_on_mac.sh
set -euo pipefail

GPU_HOST="${GPU_HOST:?请设置 GPU_HOST，例如 n194-194-195（需已在 ~/.ssh/config 配好 ProxyJump）}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# python-build-standalone（自带 OpenSSL/FFI/sqlite，目标机零编译依赖）。
# 3.10 已 EOL，20261009 是最后含 cpython-3.10 的 tag。
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.10.22%2B20261009-x86_64-unknown-linux-gnu-install_only.tar.gz"
# BtbN 静态 ffmpeg（GPU 机 apt 源无 ffmpeg，librosa 读 mp3 依赖 ffprobe）
FF_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"

echo "==> [1/4] 下载独立 Python 3.10 与静态 ffmpeg"
curl -fL --retry 3 -o "$WORK/cpython310.tar.gz" "$PY_URL"
curl -fL --retry 3 -o "$WORK/ffmpeg.tar.xz" "$FF_URL"
tar xJf "$WORK/ffmpeg.tar.xz" -C "$WORK" --wildcards '*/bin/ffmpeg' '*/bin/ffprobe'

echo "==> [2/4] 传输 Python 与 ffmpeg 到 $GPU_HOST"
scp "$WORK/cpython310.tar.gz" "$GPU_HOST":/tmp/
scp "$WORK"/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg \
    "$WORK"/ffmpeg-master-latest-linux64-gpl/bin/ffprobe "$GPU_HOST":/tmp/

echo "==> [3/4] 同步项目代码（排除虚拟环境与本地产物）"
rsync -az --delete \
  --exclude '.venv' --exclude 'output' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude '*.pyc' \
  -e ssh ./ "$GPU_HOST":/root/NewVerse/

echo "==> [4/4] 同步 HuggingFace 模型缓存（Demucs / faster-whisper / wav2vec2 对齐器）"
# torchcrepe 与 whisperx VAD 权重随 pip 包自带，无需传输
ssh "$GPU_HOST" 'mkdir -p /root/.cache/huggingface/hub'
rsync -az --partial -e ssh \
  "$HOME/.cache/huggingface/hub/" "$GPU_HOST":/root/.cache/huggingface/hub/

echo "Mac 侧物料准备完成，登录 $GPU_HOST 执行 02_install_gpu_driver.sh"
