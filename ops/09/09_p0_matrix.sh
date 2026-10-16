#!/bin/bash
# P0.2 π0 矩阵重跑 v2（09，3×A100：GPU0/1/3；GPU2 他人占用勿用）
#   Phase A（树打补丁）：SOG{spatial,object,goal}×{plain,adv} + Long-SFT libero_10×{plain,adv} = 8 块
#   Phase B（树还原）  ：SOG 三套件 original = 3 块
# v2 修复（10-16 00:20）：完成判据从"日志含 overall"（该行只写进结果文件，永不进 stdout，
#   导致无限重跑）改为"本次尝试新增 [eval] 结果 行且其文件含 overall"；已有完整结果的块自动跳过。
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
TEX=$EX/train_20261014_102227_long2/Universal_Cloth.png
LONG=$D/model/RLinf-Pi0-LIBERO-Long-SFT
LOGD=$EX/p0_logs
WD=$EX/watchdog_events.log
mkdir -p "$LOGD"
echo "$(date '+%F %T') [chain] P0 矩阵链 v2 启动 master=$$" >> "$WD"

wait_gpu_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' ')
    [ "$used" -lt 2000 ] && break
    echo "$(date '+%F %T') [chain] GPU$1 被占(${used}MiB)，等待中" >> "$WD"
    sleep 300
  done
}

have_result() {  # $1=cond $2=suite $3=期望集数：已有完整结果则跳过
  local d r n
  for d in $(ls -dt "$EX"/eval_${1}_${2}_2* 2>/dev/null); do
    r="$d/evaluation_result.txt"
    [ -f "$r" ] || continue
    n=$(grep -c '^Task:' "$r" 2>/dev/null)
    if grep -q "overall robot success" "$r" && [ "${n:-0}" -ge "$3" ]; then
      echo "$(date '+%F %T') [chain] $1 $2 已有完整结果 $(basename "$d")，跳过" >> "$WD"
      return 0
    fi
  done
  return 1
}

run_block() {  # $1=GPU $2=cond $3=suite  其余为额外参数
  local g=$1 cond=$2 suite=$3; shift 3
  local name="p0_${cond}_${suite}"
  local log="$LOGD/${name}.log"
  local before after res
  have_result "$cond" "$suite" 100 && return 0
  for attempt in 1 2 3; do
    before=$(grep -c "\[eval\] 结果" "$log" 2>/dev/null)
    echo "$(date '+%F %T') [chain] $name 第${attempt}次启动 GPU$g" >> "$WD"
    wait_gpu_free "$g"
    bash "$PI/run_cloth.sh" "$g" eval --condition "$cond" --suite "$suite" \
      --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 "$@" >> "$log" 2>&1
    after=$(grep -c "\[eval\] 结果" "$log" 2>/dev/null)
    if [ "${after:-0}" -gt "${before:-0}" ]; then
      res=$(grep "\[eval\] 结果" "$log" | tail -1 | sed "s/.*结果: //" | tr -d "\r")
      if [ -n "$res" ] && [ -f "$res" ] && grep -q "overall robot success" "$res"; then
        echo "$(date '+%F %T') [chain] $name 完成 → $res" >> "$WD"
        return 0
      fi
    fi
    echo "$(date '+%F %T') [watchdog] $name 第${attempt}次未产出完整结果，30s 后重试" >> "$WD"
    sleep 30
  done
  echo "$(date '+%F %T') [watchdog] $name 三次重试全部失败，标记 FAILED" >> "$WD"
  return 1
}

phaseA_worker() {  # $1=GPU，其余为 "cond:suite[:extra]" 列表
  local g=$1; shift
  for spec in "$@"; do
    local cond="${spec%%:*}"; local rest="${spec#*:}"
    local suite="${rest%%:*}"
    case "$rest" in
      *:*) local extra="${rest#*:}"; run_block "$g" "$cond" "$suite" $extra ;;
      *)   run_block "$g" "$cond" "$suite" ;;
    esac
  done
}

# ---------- Phase A：补丁树 ----------
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
pids=()
phaseA_worker 0 "plain:libero_spatial" "plain:libero_object" "plain:libero_goal" & pids+=($!)
phaseA_worker 1 "adv:libero_spatial:--texture $TEX" "adv:libero_object:--texture $TEX" "adv:libero_goal:--texture $TEX" & pids+=($!)
phaseA_worker 3 "plain:libero_10:--ckpt $LONG" "adv:libero_10:--texture $TEX --ckpt $LONG" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo "$(date '+%F %T') [chain] Phase A 完成，进入 Phase B（original，需还原树）" >> "$WD"

# ---------- Phase B：还原树，3 块并行 ----------
cd "$PI" && bash run_cloth.sh 0 patch --off >> "$WD" 2>&1
pids=()
phaseA_worker 0 "original:libero_spatial" & pids+=($!)
phaseA_worker 1 "original:libero_object" & pids+=($!)
phaseA_worker 3 "original:libero_goal" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

# ---------- 收尾 ----------
cd "$PI" && bash run_cloth.sh 0 patch --on >> "$WD" 2>&1
echo "$(date '+%F %T') [chain] P0 矩阵链全部完成 DONE" >> "$WD"
