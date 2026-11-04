#!/bin/bash
set -euo pipefail
cd /data/shaozhen.liu/code/tex3d
node=$1
case "$node" in 4090|09) ;; *) exit 2 ;; esac
test "$(git rev-parse HEAD)" = "$2"
root="$PWD/experiments/refcloth_stability/20261105"
controller="$PWD/ops/stability_seeds_20261104/control.py"
protocol="$PWD/ops/stability_seeds_20261105/protocol.json"
mkdir -p "$root"
exec 9>"$root/supervisor.lock"
flock -n 9 || exit 0
export PYTHONUNBUFFERED=1
guard=/dev/shm/refcloth_guard_$(id -u)
mkdir -p "$guard"
if ! pgrep -u "$USER" -f 'refcloth_resourc[e].py monitor' >/dev/null; then
    nohup setsid python3 -u ops/refcloth_resource.py monitor >>"$guard/monitor.log" 2>&1 </dev/null 9>&- &
fi
seeds=$(python3 - "$protocol" "$node" <<'PY'
import json,sys
with open(sys.argv[1]) as file:
    plan=json.load(file)['random_assignment']
print(' '.join(map(str,plan['host_41' if sys.argv[2]=='4090' else 'host_09'])))
PY
)
read -r -a assigned <<<"$seeds"
python3 -u "$controller" --protocol "$protocol" --stage prepare --node "$node" >>"$root/preparation.log" 2>&1 9>&-
if [[ "$node" == 4090 ]]; then
    python3 -u "$controller" --protocol "$protocol" --stage eval --node "$node" --resume >>"$root/eval_controller.log" 2>&1 9>&- &
    eval_pid=$!
fi
python3 -u "$controller" --protocol "$protocol" --stage source --node "$node" --resume --seeds "${assigned[@]}" >>"$root/source_controller.log" 2>&1 9>&-
python3 - "$root/train_done_$node.json" "${assigned[@]}" <<'PY'
import json,sys
with open(sys.argv[1]) as file:
    status=json.load(file)
expected={'s%s_%s'%(seed,ref) for seed in sys.argv[2:] for ref in ('native','mr')}
assert set(status['finished'])==expected,status
PY
if [[ "$node" == 4090 ]]; then
    wait "$eval_pid"
    analysis_python=/data/shaozhen.liu/envs/tex3d-oft-mj330/bin/python
    "$analysis_python" -u ops/stability_20261102/summarize.py --root "$root" --out "$root/analysis" --followup-protocol "$protocol" >>"$root/analysis.log" 2>&1 9>&-
fi
date -Is
