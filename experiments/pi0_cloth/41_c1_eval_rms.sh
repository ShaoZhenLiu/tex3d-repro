#!/bin/bash
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LOGD=$EX/main_logs
WD=$EX/watchdog_events.log
CKPT5=$D/model/RLinf-Pi05-LIBERO-130-fullshot-SFT
name=mtf_pi05_obj_c1rms
log=$LOGD/${name}.log
cell_done() {
  res=$(grep -a "\[eval\] 结果" "$log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  n=$(grep -c "^Task:" "$res" 2>/dev/null)
  [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]
}
if cell_done; then echo "$(date "+%F %T") [c1rms] 已完成跳过" >> "$WD"; exit 0; fi
echo "$(date "+%F %T") [c1rms] 补跑 GPU1（GPU3 被他人占用后 OOM）" >> "$WD"
for attempt in 1 2 3 4; do
  CLOTH_QUERY_SEED=1 CLOTH_QUERY_SEED_LOG=$EX/edpa_logs/qseed_${name}.csv \
  bash "$PI/run_cloth330.sh" 1 eval --condition adv --suite libero_object \
    --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --num-denoise 10 --init-offset 10 \
    --texture "$EX/texpush/c1_f0rms.png" --ckpt "$CKPT5" --model-config pi05_libero >> "$log" 2>&1
  res=$(grep -a "\[eval\] 结果" "$log" 2>/dev/null | tail -1 | sed "s/.*结果: //" | tr -d "\r")
  n=$(grep -c "^Task:" "$res" 2>/dev/null)
  if [ -n "$res" ] && [ -f "$res" ] && [ "${n:-0}" -ge 100 ]; then
    echo "$(date "+%F %T") [c1rms] 完成 -> $(grep -a "overall robot success" "$res" | tail -1)" >> "$WD"
    exit 0
  fi
  sleep 45
done
echo "$(date "+%F %T") [c1rms] 四次失败 FAILED" >> "$WD"
