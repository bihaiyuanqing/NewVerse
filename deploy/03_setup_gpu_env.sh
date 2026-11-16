#!/usr/bin/env bash
# 第三步（在 GPU 机上以 root 执行）：Python 3.10 环境、项目依赖与冒烟校验。
# 实测：系统 Python 仅 3.7（torch 2.6 无 cp37 wheel），KEA apt 源无 python3.10
# 与 ffmpeg，故 Python 用 01 脚本上传的 standalone 包，ffmpeg 用静态二进制。
# 用法：bash deploy/03_setup_gpu_env.sh
set -euo pipefail

PY_ROOT=/opt/python3.10
VENV=/opt/newverse-venv
PIP_INDEX="https://bytedpypi.byted.org/simple/"
PROJECT=/root/NewVerse

echo "==> [1/7] 部署独立 Python 3.10"
if [ ! -x "$PY_ROOT/bin/python3.10" ]; then
  mkdir -p "$PY_ROOT"
  tar xzf /tmp/cpython310.tar.gz -C "$PY_ROOT" --strip-components=1
fi
"$PY_ROOT/bin/python3.10" --version

echo "==> [2/7] 部署静态 ffmpeg/ffprobe"
if [ ! -x /usr/local/bin/ffprobe ]; then
  install -m 755 /tmp/ffmpeg /tmp/ffprobe /usr/local/bin/
fi
ffprobe -version | head -1

echo "==> [3/7] 创建 venv 并指向内网 PyPI"
[ -d "$VENV" ] || "$PY_ROOT/bin/python3.10" -m venv "$VENV"
"$VENV/bin/pip" config set global.index-url "$PIP_INDEX"
"$VENV/bin/pip" config set global.trusted-host bytedpypi.byted.org

echo "==> [4/7] 安装 torch 2.6.0（PyPI linux wheel 自带 cu124，R535 驱动 minor-version 兼容）"
"$VENV/bin/pip" install --no-cache-dir "torch==2.6.0" "torchaudio==2.6.0"

echo "==> [5/7] 安装项目其余依赖"
"$VENV/bin/pip" install --no-cache-dir \
  "numpy==2.2.6" "demucs==4.1.0" "whisperx==3.4.5" \
  "librosa==0.10.2.post1" "soundfile==0.13.1" "torchcrepe==0.0.24" \
  "zhconv==1.4.3" "pyyaml==6.0.3" "requests" pytest

echo "==> [6/7] 隔离安装 cuDNN 8（pyannote/onnxruntime 需要，与 torch 的 cuDNN 9 共存）"
# 直接装进 site-packages 会把 torch 锁定的 9.1.0.70 降级覆盖，必须 --target 隔离
"$VENV/bin/pip" install --no-cache-dir --target=/opt/nvidia-cudnn8 \
  "nvidia-cudnn-cu12==8.9.7.29"
"$VENV/bin/pip" install --no-cache-dir "nvidia-cudnn-cu12==9.1.0.70"
NV_LIBS="$(echo "$VENV"/lib/python3.10/site-packages/nvidia/*/lib | tr ' ' ':')"
cat > "$PROJECT/gpu_env.sh" <<EOF
# NewVerse GPU 运行环境（由 03_setup_gpu_env.sh 生成），运行前 source
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="/opt/nvidia-cudnn8/nvidia/cudnn/lib:$NV_LIBS:\$LD_LIBRARY_PATH"
EOF

echo "==> [7/7] 冒烟校验"
cd "$PROJECT"
# shellcheck disable=SC1091
source gpu_env.sh
python - <<'PY'
import torch, ctranslate2
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
print("CUDA 设备:", torch.cuda.device_count(), "张", torch.cuda.get_device_name(0))
print("CTranslate2 CUDA:", ctranslate2.get_cuda_device_count(), "设备")
import demucs, whisperx, torchcrepe, librosa  # noqa
print("P0 组件全部导入成功")
PY
python -m pytest tests/ -q
echo "环境就绪：source $PROJECT/gpu_env.sh 后即可运行 cli.py"
