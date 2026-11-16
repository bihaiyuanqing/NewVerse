#!/usr/bin/env bash
# GPU 机：创建 YingMusic 独立 venv 并安装推理精简依赖。
# 与 /opt/newverse-venv（P0）完全隔离，避免 numba/transformers 版本冲突。
# 用法（在 GPU 机上）：bash /root/NewVerse/deploy/04_setup_yingmusic_venv.sh
set -euo pipefail

PYTHON_BIN=/opt/python3.10/bin/python3
VENV=/opt/yingmusic-venv
REPO=/root/YingMusic-Singer-Plus
REQ=/root/NewVerse/deploy/yingmusic-requirements.txt
PIP_INDEX=http://bytedpypi.byted.org/simple/

echo "==> 创建独立 venv：${VENV}"
"${PYTHON_BIN}" -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -U "pip<25" -i "${PIP_INDEX}"

echo "==> 安装推理精简依赖（内网源）"
"${VENV}/bin/pip" install -r "${REQ}" -i "${PIP_INDEX}"

echo "==> 离线导入冒烟（不加载权重，只验证依赖闭合）"
cd "${REPO}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${VENV}/bin/python" - <<'PY'
import hydra, einops, ema_pytorch, omegaconf
import torchdiffeq, vector_quantize_pytorch, x_transformers, alias_free_torch
import dac, phonemizer, onnxruntime, cn2an, jieba, pypinyin
from transformers import BertTokenizer
from src.YingMusicSinger.infer.YingMusicSinger import YingMusicSinger
print("import smoke OK")
PY

echo "==> 完成。运行重唱前先 source /root/NewVerse/gpu_env.sh"
