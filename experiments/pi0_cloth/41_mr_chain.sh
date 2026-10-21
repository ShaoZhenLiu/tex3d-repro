#!/bin/bash
set -u
EX=/data/shaozhen.liu/code/tex3d/experiments/pi0_cloth
WD=$EX/watchdog_events.log
D=/data/shaozhen.liu
cd $D/code/tex3d-ws/g7
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TMPDIR=$D/tmp PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
env CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 $ENVV \
  $D/envs/tex3d-oft-mj330/bin/python \
  $D/code/tex3d/openvla-oft/experiments/robot/libero/attack_oft_cloth_tvfix_mr.py train \
  --pool-tasks 0,2,6,8 --pool-file $EX/oft_cloth/r0f_pool.pt --load-pool \
  --iters 5000 --seed 7 --uv-eot --batch 8 --alpha-action 0.0 --alpha-feature 10.0 \
  --mr-cache $EX/oft_cloth/mr_plain_cache.pt --checkpoint-every 625 \
  --tag tvfix-mr >> $EX/edpa_logs/train_tvfix_mr.log 2>&1
echo "$(date "+%F %T") [mr] 训练结束 → 启动首轮评测（obj@GPU3 sp@GPU2）" >> "$WD"
bash $EX/41_tvfix_eval.sh mr 3 2
