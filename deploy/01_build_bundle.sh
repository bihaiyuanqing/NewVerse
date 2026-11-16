#!/usr/bin/env bash
# 在有外网的 Mac 上执行：把 GPU 机需要的全部依赖、代码、权重下载到本地 bundle。
# 用法：bash deploy/01_build_bundle.sh
set -euo pipefail

# ============ 按需修改的变量 ============
BUNDLE_DIR="${BUNDLE_DIR:-$HOME/newverse_offline_bundle}"
GPU_PY_VERSION="3.10"                 # YingMusic-Singer-Plus 官方要求 3.10
CUDA_INDEX="https://download.pytorch.org/whl/cu121"
YING_REPO="https://github.com/ASLP-lab/YingMusic-Singer-Plus.git"
# ========================================

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$BUNDLE_DIR"/{wheelhouse,models,repos,cache_torch,cache_hf}

echo "==> [1/5] 克隆 YingMusic-Singer-Plus 仓库代码"
if [ ! -d "$BUNDLE_DIR/repos/YingMusic-Singer-Plus/.git" ]; then
  git clone --depth 1 "$YING_REPO" "$BUNDLE_DIR/repos/YingMusic-Singer-Plus"
fi

echo "==> [2/5] 下载 Linux x86_64 / Python ${GPU_PY_VERSION} wheel（torch 走 CUDA index）"
# 项目自身依赖（torch/torchaudio 指定 cu121）
uv pip download --python-platform x86_64-manylinux_2_28 --python-version "${GPU_PY_VERSION}" \
  --index-url "$CUDA_INDEX" --index-strategy unsafe-best-match \
  -d "$BUNDLE_DIR/wheelhouse" torch torchaudio
uv pip download --python-platform x86_64-manylinux_2_28 --python-version "${GPU_PY_VERSION}" \
  -d "$BUNDLE_DIR/wheelhouse" -r requirements.txt
# YingMusic 官方依赖
uv pip download --python-platform x86_64-manylinux_2_28 --python-version "${GPU_PY_VERSION}" \
  -d "$BUNDLE_DIR/wheelhouse" \
  -r "$BUNDLE_DIR/repos/YingMusic-Singer-Plus/requirements.txt" || \
  echo "[警告] 部分包无 manylinux wheel，需在 03 脚本中按报错单独补 sdist 构建"

echo "==> [3/5] 预下载模型权重（HuggingFace）"
# 重唱模型（含 VAE 等全部权重）
huggingface-cli download ASLP-lab/YingMusic-Singer-Plus \
  --local-dir "$BUNDLE_DIR/models/YingMusic-Singer-Plus"
# WhisperX 转写模型与中文对齐模型
huggingface-cli download Systran/faster-whisper-small \
  --local-dir "$BUNDLE_DIR/models/faster-whisper-small"
huggingface-cli download jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
  --local-dir "$BUNDLE_DIR/models/wav2vec2-xlsr-chinese"

echo "==> [4/5] 预热 torch hub 缓存（demucs 权重、torchcrepe 权重走 github/fbaipublicfiles）"
# 在 Mac 上各跑一次触发下载，再把缓存整体打包
.venv/bin/python - <<'PY'
import torch
from demucs.pretrained import get_model
get_model("htdemucs_ft")
import torchcrepe, torch
torchcrepe.predict(torch.zeros(1, 16000), 16000, hop_length=160, model="full",
                   device="cpu", decoder=torchcrepe.decode.viterbi)
print("torch hub 缓存预热完成")
PY
cp -R "$HOME/.cache/torch/hub/checkpoints/." "$BUNDLE_DIR/cache_torch/" 2>/dev/null || true
cp -R "$HOME/.cache/huggingface/." "$BUNDLE_DIR/cache_hf/" 2>/dev/null || true

echo "==> [5/5] 打包项目代码"
tar --exclude='.venv' --exclude='output' --exclude='.git' \
  -czf "$BUNDLE_DIR/newverse_project.tar.gz" -C "$(dirname "$0")/.." .

echo "完成。bundle 位置：$BUNDLE_DIR"
du -sh "$BUNDLE_DIR"
