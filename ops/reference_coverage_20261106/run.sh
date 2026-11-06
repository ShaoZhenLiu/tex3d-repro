#!/bin/bash
set -euo pipefail
cd /data/shaozhen.liu/code/tex3d
node=$1
commit=$2
case "$node" in 4090|09) ;; *) exit 2 ;; esac
test "$(git rev-parse HEAD)" = "$commit"
root="$PWD/experiments/refcloth_reference_coverage/20261106"
exec 9>"$root/supervisor_$node.lock"
flock -n 9 || exit 0
export PYTHONPATH="$PWD/ops"
export PYTHONUNBUFFERED=1
guard=/dev/shm/refcloth_guard_$(id -u)
mkdir -p "$guard"
if ! pgrep -u "$USER" -f 'refcloth_resourc[e].py monitor' >/dev/null; then
    nohup setsid python3 -u ops/refcloth_resource.py monitor >>"$guard/monitor.log" 2>&1 </dev/null 9>&- &
fi
exec python3 -u ops/reference_coverage_20261106/service.py --node "$node" --commit "$commit"
