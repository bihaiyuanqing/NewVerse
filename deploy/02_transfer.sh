#!/usr/bin/env bash
# 把离线 bundle 经 SSH 传到 GPU 机。
# 用法：GPU_HOST=user@1.2.3.4 GPU_PORT=2222 GPU_PATH=/data/newverse bash deploy/02_transfer.sh
set -euo pipefail

GPU_HOST="${GPU_HOST:?请设置 GPU_HOST，例如 user@1.2.3.4}"
GPU_PORT="${GPU_PORT:-22}"
GPU_PATH="${GPU_PATH:-/data/newverse}"
BUNDLE_DIR="${BUNDLE_DIR:-$HOME/newverse_offline_bundle}"

echo "==> 传输到 ${GPU_HOST}:${GPU_PATH}（端口 ${GPU_PORT}）"
ssh -p "$GPU_PORT" "$GPU_HOST" "mkdir -p '$GPU_PATH'"
rsync -avz --progress -e "ssh -p ${GPU_PORT}" \
  "$BUNDLE_DIR" "$GPU_HOST:$GPU_PATH/"

echo "传输完成。登录 GPU 机后执行："
echo "  cd $GPU_PATH/$(basename "$BUNDLE_DIR") && bash 03_offline_install.sh"
