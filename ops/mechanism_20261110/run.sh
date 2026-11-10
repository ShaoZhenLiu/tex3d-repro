#!/bin/bash
set -euo pipefail
cd /data/shaozhen.liu/code/tex3d
commit=$1
test "$(git rev-parse HEAD)" = "$commit"
root="$PWD/experiments/refcloth_mechanism/20261110"
mkdir -p "$root"
exec 9>"$root/controller.lock"
flock -n 9 || exit 0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/ops:$PWD/ops/mechanism_20261110"
guard=/dev/shm/refcloth_guard_$(id -u)
mkdir -p "$guard"
if ! pgrep -u "$USER" -f 'refcloth_resourc[e].py monitor' >/dev/null; then
    nohup setsid python3 -u ops/refcloth_resource.py monitor >>"$guard/monitor.log" 2>&1 </dev/null 9>&- &
fi
exec python3 -u ops/mechanism_20261110/control.py --commit "$commit"
