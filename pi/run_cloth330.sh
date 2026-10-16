#!/bin/bash
# 桌布实验统一启动器：bash run_cloth.sh <GPU_ID> <子命令...>
# 例：bash run_cloth.sh 1 smoke
#     bash run_cloth.sh 1 train --iters 5000
#     bash run_cloth.sh 2 eval --condition adv --texture <png> --tasks heldout
set -eu
GPU="${1:?用法: run_cloth.sh <GPU_ID> <args...>}"; shift

export PATH=/opt/rh/devtoolset-9/root/usr/bin:/usr/local/cuda-12.1/bin:$PATH
export CC=/opt/rh/devtoolset-9/root/usr/bin/gcc
export CXX=/opt/rh/devtoolset-9/root/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR=/data6/shaozhen.liu/tex3d-mig/torch_extensions

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=/data/shaozhen.liu/.cache/openpi
export MPLCONFIGDIR=/tmp/attack_pi_mpl
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/shaozhen.liu/code/tex3d-deps/taming-transformers${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=/data/shaozhen.liu/envs/tex3d-pi0-mj330/bin/python
TORCH_LIB_DIR=$(dirname "$PYTHON_BIN")/../lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$PYTHON_BIN" /data/shaozhen.liu/code/tex3d/pi/attack_pi0_cloth.py "$@"
