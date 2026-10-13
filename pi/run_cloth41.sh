#!/bin/bash
# 4090-48G 专用启动器（41 无 /data6，无 devtoolset 需求；nvdiffrast 扩展沿用 41 历史缓存）
# 用法：bash run_cloth41.sh <GPU_ID> <子命令...>
set -eu
GPU="${1:?用法: run_cloth41.sh <GPU_ID> <args...>}"; shift

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=/data/shaozhen.liu/.cache/openpi
export MPLCONFIGDIR=/tmp/attack_pi_mpl
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/shaozhen.liu/code/tex3d-deps/taming-transformers${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=/data/shaozhen.liu/envs/tex3d-pi0/bin/python
TORCH_LIB_DIR=$(dirname "$PYTHON_BIN")/../lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$PYTHON_BIN" /data/shaozhen.liu/code/tex3d/pi/attack_pi0_cloth.py "$@"
