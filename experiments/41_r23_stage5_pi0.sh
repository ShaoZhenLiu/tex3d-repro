#!/bin/bash
# 41_r23_stage5_wave.sh — 阶段⑤π0波：π0.5 exposure 120 回合（GPU3/7 分套件）+ traj 18 回合（GPU5）
set -u
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/r23_stage5
LOGD=$EX/logs; mkdir -p "$LOGD"
CKPT5=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
TEXMR=$D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png
TEXRND=$D/code/tex3d/experiments/pi0_cloth/r19_random_ctrl/textures/random_cloth_s101.png
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOGD/master.log"; }
mem_free() { nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
io_calm() { local l=$(vmstat 3 2 | tail -1); local bi=$(echo $l | awk "{print \$9}"); local wa=$(echo $l | awk "{print \$16}"); [ "$bi" -lt 200000 ] && [ "$wa" -lt 10 ]; }

run_exp() {  # $1=gpu $2=suites
  local g=$1 ss=$2 w=0
  while [ "$(mem_free "$g")" -lt 22000 ]; do sleep 180; w=$((w+1)); [ $w -gt 40 ] && return 1; done
  while ! io_calm; do sleep 120; done
  log "exp[$ss] GPU$g 启动"
  cd $PI && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=$D/.cache/openpi MPLCONFIGDIR=$D/tmp/attack_pi_mpl PYTHONUNBUFFERED=1 \
    $D/envs/tex3d-pi0-mj330/bin/python eval_cloth_instrumented.py exposure \
    --ckpt "$CKPT5" --model-config pi0_libero --num-denoise 10 \
    --out $EX/exp_pi0 --limit-suites "$ss" >> "$LOGD/exp_$ss.log" 2>&1
  log "exp[$ss] GPU$g 结束: $(ls $EX/exp_pi0/exp_${ss//,/_}_*.json 2>/dev/null | wc -l) 记录"
}
run_traj() {
  local g=$1 w=0
  while [ "$(mem_free "$g")" -lt 22000 ]; do sleep 180; w=$((w+1)); [ $w -gt 40 ] && return 1; done
  while ! io_calm; do sleep 120; done
  log "traj GPU$g 启动"
  cd $PI && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=$D/.cache/openpi MPLCONFIGDIR=$D/tmp/attack_pi_mpl PYTHONUNBUFFERED=1 \
    $D/envs/tex3d-pi0-mj330/bin/python eval_cloth_instrumented.py traj \
    --ckpt "$CKPT5" --model-config pi0_libero --num-denoise 10 \
    --out $EX/traj_pi0 --tex-mr "$TEXMR" --tex-random "$TEXRND" >> "$LOGD/traj.log" 2>&1
  log "traj GPU$g 结束: $(ls $EX/traj_pi0/traj_*.json 2>/dev/null | wc -l)/18 记录"
}
( run_exp 3 spatial,goal ) &
sleep 300
( run_exp 7 object,10 ) &
sleep 300
( run_traj 5 ) &
log "阶段⑤π0波拉起（exp×2 + traj×1，300s 错峰 + IO 门控）"
wait
log "阶段⑤ π0.5 波收口：exp=$(ls $EX/exp_pi0/exp_*.json 2>/dev/null | wc -l)/120 traj=$(ls $EX/traj_pi0/traj_*.json 2>/dev/null | wc -l)/18"
