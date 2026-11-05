#!/bin/bash
set -euo pipefail
cd /data/shaozhen.liu/code/tex3d
test "$(git rev-parse HEAD)" = "$1"
root="$PWD/experiments/refcloth_stability/20261105"
exec 9>"$root/evaluation09_supervisor.lock"
flock -n 9 || exit 0
controller="$PWD/ops/stability_seeds_20261104/control.py"
protocol="$PWD/ops/stability_seeds_20261105/protocol.json"
until python3 -u "$controller" --protocol "$protocol" --stage eval --node 09 --resume >>"$root/eval_controller09.log" 2>&1 9>&-; do
    printf 'A100_EVAL_RETRY %s\n' "$(date -Is)"
    sleep 60
done
