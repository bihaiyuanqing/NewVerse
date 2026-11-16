#!/usr/bin/env bash
# 在无外网的 GPU 机上执行：离线建环境、装依赖、校验。
# 用法：bash 03_offline_install.sh
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$PWD/newverse_app"

echo "==> [1/4] 解压项目代码"
mkdir -p "$PROJECT_DIR"
tar -xzf "$BUNDLE_DIR/newverse_project.tar.gz" -C "$PROJECT_DIR"

echo "==> [2/4] 创建 Python 3.10 venv（机器上需已有 python3.10）"
python3.10 -m venv "$PROJECT_DIR/.venv"
# shellcheck disable=SC1091
source "$PROJECT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip

echo "==> [3/4] 离线安装全部 wheel"
pip install --no-index --find-links "$BUNDLE_DIR/wheelhouse" \
  -r "$PROJECT_DIR/requirements.txt"
pip install --no-index --find-links "$BUNDLE_DIR/wheelhouse" \
  -r "$BUNDLE_DIR/repos/YingMusic-Singer-Plus/requirements.txt"

echo "==> [4/4] 还原模型缓存并写入离线环境变量"
mkdir -p "$HOME/.cache/torch/hub/checkpoints" "$HOME/.cache/huggingface"
cp -Rn "$BUNDLE_DIR/cache_torch/." "$HOME/.cache/torch/hub/checkpoints/" 2>/dev/null || true
cp -Rn "$BUNDLE_DIR/cache_hf/." "$HOME/.cache/huggingface/" 2>/dev/null || true

cat > "$PROJECT_DIR/offline_env.sh" <<EOF
# 离线运行环境变量，启动前 source
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=$HOME/.cache/huggingface
export NEWVERSE_YING_REPO=$BUNDLE_DIR/repos/YingMusic-Singer-Plus
export NEWVERSE_YING_CKPT=$BUNDLE_DIR/models/YingMusic-Singer-Plus
EOF

echo "==> 冒烟校验"
source "$PROJECT_DIR/offline_env.sh"
python - <<'PY'
import torch
from core import device
print("CUDA 可用:", torch.cuda.is_available())
print("设备决策:", device.describe())
import demucs, whisperx, torchcrepe  # noqa
print("P0 组件导入成功")
PY

echo "离线安装完成。项目目录：$PROJECT_DIR"
