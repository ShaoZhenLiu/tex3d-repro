#!/bin/bash
# 41_r18_e02_oft.sh — 第18轮 E02：OFT 源端(Spatial step150000)闭环 四条件×100回合
#   条件 O/P/N8/MR 全部新增：历史 R1 O/P 为初态0-9、300步（transfer_pi2oft 旧默认），
#   与本协议（初态10-19、400步+10初始化步）不符，不复用。
#   入口 = transfer_pi2oft.py 扩展版（--max-steps、结果头登记 init_offset/ckpt/max_steps/
#   texture_sha256）；cwd=g7；tex3d-oft-mj330；OFT 原生动作链（不用 π 入口加载 OFT）。
#   场景 XML 并发约束（OFT 树与 π 树独立）：plain/n8/mr 同为 patch-on 可并行；
#   original(patch-off) 必须等三个 patch-on 条件全部 100 回合收口后才能启动（共享 XML）。
#   GPU 纪律（2026-10-26 用户授权更新：除 GPU4 外全部可用）：候选 0/1/2/3/5/6/7、
#   每卡≤2 我方进程（与 E01 共用 flock 准入/claim 目录）、VRAM 守卫 need≈20000MiB
#   （OFT 实测≈16.4-18G）、60s 错峰、重试×4、动态选卡。GPU4 永不使用。
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=/data/shaozhen.liu
EX=$D/code/tex3d/experiments/r18_e01e02
LOGD=$EX/e02_logs
CLAIMD=$EX/gpu_claims
LOCK=$EX/gpu_admit.lock
WD=$D/code/tex3d/experiments/pi0_cloth/watchdog_events.log
SCRIPT=$D/code/tex3d/openvla-oft/experiments/robot/libero/transfer_pi2oft.py
PY=$D/envs/tex3d-oft-mj330/bin/python
TEXN8=$D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_153634_tvfix-bs8/Universal_Cloth_OV.png
TEXMR=$D/code/tex3d/experiments/pi0_cloth/oft_cloth/train_20261021_224922_tvfix-mr/Universal_Cloth_OV.png
T0=$D/code/tex3d-deps/libero/libero/libero/assets/tablecloth/cloth_base.png
mkdir -p "$LOGD" "$CLAIMD"
check_sha() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "SHA 不符: $1" >> "$WD"; exit 1; }; }
check_sha "$TEXN8" ee241502c8d1523ae3fcb3b038304ff77c22772f5c0b4e56f7aa0a0bf71509ec
check_sha "$TEXMR" 51ad185f0495373784fcf39b48cdbcf75b32b947cdaabf71429e1868de863187
check_sha "$T0"    9829a9529b74f02e61a87ee27c9acceddef9fed98a45c8bc829b8c233bda43c3
ENVV="MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$D/code/tex3d:$D/code/tex3d/openvla-oft"
log() { echo "$(date '+%F %T') [e02] $*" | tee -a "$WD"; }

gpu_used_mi() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }
release_claim() { rm -f "$CLAIMD/$1".* 2>/dev/null; }
pick_gpu() {  # $1=泳道名 $2=需要空闲MiB → 成功输出 GPU id（claim 已登记）
  # 准入规则：该卡已有 claim 的需求之和 + 自身需求 ≤ 当前空闲，且 claim 数<2
  # claim 记泳道壳 pid（由调用方 run_cell/run_cond 传入；命令替换子壳会即刻退出）
  local name=$1 need=$2 lane_pid=$3 tries=0 g used free cnt f pid ned sum_need out
  while [ $tries -lt 2880 ]; do
    out=$(
      (
        flock 9
        for g in 5 7 6 2 1 0 3; do
          sum_need=0; cnt=0
          for f in "$CLAIMD"/*."$g"; do
            [ -e "$f" ] || continue
            read -r pid ned < "$f" 2>/dev/null || { rm -f "$f"; continue; }
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
            sum_need=$((sum_need + ${ned:-10000})); cnt=$((cnt + 1))
          done
          [ "$cnt" -ge 2 ] && continue
          used=$(gpu_used_mi "$g")
          [ -z "$used" ] && continue
          free=$((49140 - used))
          if [ $((sum_need + need)) -le "$free" ]; then
            echo "$lane_pid $need" > "$CLAIMD/${name}.$g"
            echo "$g"
            exit 0
          fi
        done
        exit 1
      ) 9>"$LOCK"
    ) && { echo "$out"; return 0; }
    sleep 15; tries=$((tries + 1))
  done
  return 1
}
cell_done() {
  local res n
  res=$(grep -a '\[eval\] 结果' "$LOGD/$1.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  n=$(grep -ac '^Task:' "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
run_cond() {  # $1=name $2=cond $3=texture(可空)
  local name=$1 cond=$2 tex=$3 res n g targs=() mypid=$BASHPID
  if cell_done "$name"; then log "评测 $name 已完成，跳过"; return 0; fi
  [ -n "$tex" ] && targs=(--texture "$tex")
  local log_f="$LOGD/${name}.log"
  for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    release_claim "$name"
    g=$(pick_gpu "$name" 20000 "$mypid") || { log "评测 $name 等待 GPU 空闲超时 FAILED"; return 1; }
    log "评测 $name 尝试 $attempt GPU$g"
    ( cd $D/code/tex3d-ws/g7 && env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g $ENVV \
      $PY $SCRIPT eval --condition "$cond" --suite libero_spatial \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --init-offset 10 --max-steps 400 \
      ${targs[@]+"${targs[@]}"} >> "$log_f" 2>&1 )
    res=$(grep -a '\[eval\] 结果' "$log_f" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
    n=$(grep -ac '^Task:' "$res" 2>/dev/null)
    if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
      release_claim "$name"
      log "评测 $name 完成 -> $(grep -a 'overall robot success' "$res" | tail -1)"
      return 0
    fi
    sleep 120
  done
  release_claim "$name"
  log "评测 $name 四次失败 FAILED"; return 1
}

# patch-on 三条件并行（60s 错峰；GPU 动态准入，claim 与 E01 共享保证≤2/卡）
pids=()
( sleep 0;   run_cond e02_plain   plain "" ) & pids+=($!)
( sleep 60;  run_cond e02_native8 adv  "$TEXN8" ) & pids+=($!)
( sleep 120; run_cond e02_mr8     adv  "$TEXMR" ) & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

# original(patch-off) 必须最后：等三个 patch-on 条件收口（上限 6h，超时则告警仍继续）
t=0
until [ $t -ge 360 ]; do
  cell_done e02_plain && cell_done e02_native8 && cell_done e02_mr8 && break
  sleep 60; t=$((t + 1))
done
cell_done e02_plain && cell_done e02_native8 && cell_done e02_mr8 \
  || log "警告：patch-on 三条件未全部收口即启动 original"
run_cond e02_original original ""

ok=0; sum=""
for name in e02_plain e02_native8 e02_mr8 e02_original; do
  cell_done "$name" && ok=$((ok + 1))
  res=$(grep -a '\[eval\] 结果' "$LOGD/$name.log" 2>/dev/null | tail -1 | sed 's/.*结果: //' | tr -d '\r')
  sum="$sum$name=$(grep -a 'overall robot success' "$res" 2>/dev/null | tail -1 | grep -o '[0-9]*/100') "
done
if [ "$fail" = 0 ] && [ "$ok" = 4 ]; then
  echo "DONE-E02-OFT 4/4 $(date)：$sum" >> "$LOGD/summary.txt"; log "DONE-E02-OFT 4/4：$sum"
else
  echo "E02-OFT-WITH-FAILURES ok=$ok/4 $(date)：$sum" >> "$LOGD/summary.txt"; log "E02-OFT-WITH-FAILURES ok=$ok/4：$sum"
fi
