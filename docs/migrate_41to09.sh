#!/bin/bash
# Tex3D 迁移 41→09，2026-10-12
# 分段顺序：code小件(16G) → uv解释器(1G) → venv(26G) → 工作区(230G) → 模型(80G)
# 设计：前四段完成即可开始补跑 6 个评测（它们只用 ws 内 ckpt）；模型段最后跑不影响
# 日志：本目录 migrate_41to09.log；可重复执行（rsync 增量续传）
LOG=/data/shaozhen.liu/code/tex3d-deps/migrate_41to09.log
DST=shaozhen.liu@172.18.33.9
BASE=/data6/shaozhen.liu/tex3d-mig
SSHOPTS="-o BatchMode=yes -o ConnectTimeout=20"

ION=""
command -v ionice >/dev/null 2>&1 && ION="ionice -c3"

seg() {
  local name=$1; shift
  echo "[$(date '+%F %T')] === SEGMENT $name START ==="
  "$@"
  local rc=$?
  echo "[$(date '+%F %T')] === SEGMENT $name END rc=$rc ==="
}

{
echo "[$(date '+%F %T')] migrate_41to09.sh launched (pid $$)"

seg code_small $ION rsync -a --stats -e "ssh $SSHOPTS" \
  /data/shaozhen.liu/code/tex3d \
  /data/shaozhen.liu/code/tex3d-deps \
  /data/shaozhen.liu/code/openpi \
  $DST:$BASE/code/

seg uv_python $ION rsync -a --stats -e "ssh $SSHOPTS" \
  /home/shaozhen.liu/.local/share/uv/python \
  $DST:$BASE/uv-python/

seg envs $ION rsync -a --stats -e "ssh $SSHOPTS" \
  /data/shaozhen.liu/envs \
  $DST:$BASE/

seg ws $ION rsync -a --stats -e "ssh $SSHOPTS" \
  /data/shaozhen.liu/code/tex3d-ws \
  $DST:$BASE/code/

seg model $ION rsync -a --stats -e "ssh $SSHOPTS" \
  /data/shaozhen.liu/model/openvla-7b-finetuned-libero-spatial \
  /data/shaozhen.liu/model/openvla-7b-finetuned-libero-object \
  /data/shaozhen.liu/model/openvla-7b-oft-finetuned-libero-spatial \
  /data/shaozhen.liu/model/openvla-7b-oft-finetuned-libero-object \
  /data/shaozhen.liu/model/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
  /data/shaozhen.liu/model/RLinf-Pi0-LIBERO-Long-SFT \
  /data/shaozhen.liu/model/RLinf-Pi05-LIBERO-SFT \
  $DST:$BASE/model/

echo "[$(date '+%F %T')] ALL_SEGMENTS_DONE"
} >> "$LOG" 2>&1
