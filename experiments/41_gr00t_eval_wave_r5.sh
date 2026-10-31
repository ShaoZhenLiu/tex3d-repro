#!/bin/bash
# 41_gr00t_eval_wave.sh — GR00T 桌布评测波（4条件×4套件×100回合=1600回合）
# 用法： bash 41_gr00t_eval_wave.sh <GPU_ID> <suite1,suite2>
set -u
D=/data/shaozhen.liu
GPU=$1; SUITES=$2
V=$D/code/isaac-gr00t/.venv/bin/python
CODE=$D/code/isaac-gr00t
MODELS=/data6/shaozhen.liu/models/GR00T-N1.7-LIBERO
EX=$D/code/tex3d/experiments/r24_groot_eval
LOGD=$EX/logs; DONED=$LOGD/done
mkdir -p "$LOGD" "$DONED" "$EX/results"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOGD/master_gpu$GPU.log"; }

export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$CODE:$CODE/src

TEX_T0="$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth/cloth_base.png"
TEX_NAT="$D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_153634_tvfix-bs8/Universal_Cloth_OV.png"
TEX_MR="$D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png"

run_cell() {  # $1=name $2=suite $3=ckpt_dir $4=condition $5=texture
  local name=$1 suite=$2 ckpt=$3 cond=$4 tex=$5
  local out="$EX/results/${name}.txt"
  [ -f "$DONED/$name.done" ] && { log "$name skip"; return 0; }
  local extra=""
  [ -n "$tex" ] && extra="--texture $tex"
  for att in 1 2 3; do
    local free=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -lt 15000 ] && { sleep 300; continue; }
    log "$name att$att GPU$GPU free=$free"
    cd /tmp && timeout 7200 $V $CODE/eval_gr00t_final_r5.py \
      --ckpt-dir "$ckpt" --suite "$suite" --condition "$cond" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --init-offset 10 \
      --max-steps 400 --device cuda:0 $extra --out "$out" \
      >> "$LOGD/${name}.log" 2>&1
    local n=$(grep -ac "^Task:" "$out" 2>/dev/null)
    if [ "${n:-0}" -ge 100 ]; then
      echo "$out" > "$DONED/$name.done"
      log "$name DONE $(grep -a 'overall' "$out" | tail -1)"
      return 0
    fi
    sleep 120
  done
  log "$name FAILED"
}

# 逐套件逐条件
for suite in $(echo $SUITES | tr ',' ' '); do
  # 映射套件名到目录名
  case $suite in
    spatial) ckpt_dir="$MODELS/libero_spatial"; suite_arg="libero_spatial"; short="sp";;
    object)  ckpt_dir="$MODELS/libero_object";  suite_arg="libero_object";  short="obj";;
    goal)    ckpt_dir="$MODELS/libero_goal";    suite_arg="libero_goal";    short="go";;
    10)      ckpt_dir="$MODELS/libero_10";      suite_arg="libero_10";      short="10";;
  esac

  # 4 条件顺序执行（original 先跑做基线）
  run_cell "groot_orig_${short}"  "$suite_arg" "$ckpt_dir" "original" ""
  run_cell "groot_plain_${short}" "$suite_arg" "$ckpt_dir" "plain" "$TEX_T0"
  run_cell "groot_nat_${short}"   "$suite_arg" "$ckpt_dir" "adv" "$TEX_NAT"
  run_cell "groot_mr_${short}"    "$suite_arg" "$ckpt_dir" "adv" "$TEX_MR"
done

log "GPU$GPU 全部收口 done=$(ls $DONED/ | grep -c groot) /16"
