#!/bin/bash
# jobq_lib.sh — 作业函数库（被 jobq 作业行 source）；每个函数含产物校验，失败返回非零
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
OFT=$D/code/tex3d/openvla-oft/experiments/robot/libero
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/r3_logs
STAGE=$D/code/tex3d-deps/cloth_variants
LIVE_PI=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
LIVE_OFT=$D/code/tex3d-deps/libero/libero/libero/assets/tablecloth
CKPT=$D/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
POOL=$EX/cloth_pool_fs6040.pt
mkdir -p "$LOGD"

pi_train() {  # $1=GPU $2=tag $3=lms $4=lcon $5=ltask $6=uv(on|off)
  local g=$1 tag=$2 lms=$3 lcon=$4 ltask=$5 uv=$6 uvflag=""
  [ "$uv" = on ] && uvflag="--uv-eot"
  bash "$PI/run_cloth330.sh" "$g" train --suite libero_spatial --pool-tasks 0,2,6,8 \
    --pool-mode full --frame-weighting taao --epsilon 0.2510 --seed 7 --iters 5000 \
    --load-pool --pool-file "$POOL" $uvflag --ckpt "$CKPT" \
    --lambda-mse "$lms" --lambda-con "$lcon" --lambda-task "$ltask" --tag "$tag" \
    > "$LOGD/train_$tag.log" 2>&1
  ls -t $EX/train_*_$tag/Universal_Cloth.png 2>/dev/null | head -1
}

pi_eval() {  # $1=GPU $2=name $3=cond $4=tasks $5=trials $6=nd $7=texture(optional)
  local g=$1 name=$2 cond=$3 tasks=$4 trials=$5 nd=$6 tex=${7:-}
  local log="$LOGD/$name.log" extra=() res
  [ -n "$tex" ] && extra=(--texture "$tex")
  bash "$PI/run_cloth330.sh" "$g" eval --ckpt "$CKPT" --condition "$cond" \
    --suite libero_spatial --tasks "$tasks" --trials "$trials" --num-denoise "$nd" \
    "${extra[@]}" >> "$log" 2>&1
  res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"
}

oft_eval() {  # $1=GPU $2=name $3=cond $4=trials $5=texture(optional)
  local g=$1 name=$2 cond=$3 trials=$4 tex=${5:-}
  local log="$LOGD/$name.log" extra=() res
  [ -n "$tex" ] && extra=(--texture "$tex")
  local ENVV="CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
  ( cd $D/code/tex3d-ws/g7 && env $ENVV $D/envs/tex3d-oft-mj330/bin/python \
      "$OFT/transfer_pi2oft.py" eval --condition "$cond" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials "$trials" "${extra[@]}" ) >> "$log" 2>&1
  res=$(grep -a "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"
}
