#!/bin/bash
set -euo pipefail
cd /data/shaozhen.liu/code/tex3d
root="$PWD/experiments/refcloth_uv_gradient/20261109"
mkdir -p "$root/logs"
exec 9>"$root/supervisor.lock"
flock -n 9 || exit 0
export PYTHONPATH="$PWD/ops:$PWD/ops/uv_gradient_20261109"
export PYTHONUNBUFFERED=1
guard=/dev/shm/refcloth_guard_$(id -u)
mkdir -p "$guard"
if ! pgrep -u "$USER" -f 'refcloth_resourc[e].py monitor' >/dev/null; then
    nohup setsid python3 -u ops/refcloth_resource.py monitor >>"$guard/monitor.log" 2>&1 </dev/null 9>&- &
fi
exec python3 -u ops/uv_gradient_20261109/control.py
