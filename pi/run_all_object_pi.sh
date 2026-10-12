#!/bin/sh
set -eu

# Run all 10 LIBERO-Object tasks sequentially with the correct object/task pair.
# Extra arguments are forwarded to attack_pi.py through run_attack_pi.sh.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUN_ONE="$SCRIPT_DIR/run_attack_pi.sh"
TEXTURE_ROOT=${PI_OBJECT_TEXTURE_ROOT:-/data/shaozhen.liu/code/tex3d/experiments/pi0_attacks}

usage() {
    echo "Usage: $0 {train|adv_test|clean_test|train_and_test} [attack_pi options]"
    echo ""
    echo "Examples:"
    echo "  ATTACK_GPU_ID=7 $0 train --attack_iters 5000"
    echo "  ATTACK_GPU_ID=7 $0 adv_test --eval_num_trials 50 --replan_steps 5"
    echo "  ATTACK_GPU_ID=7 $0 clean_test --eval_num_trials 50 --replan_steps 5"
    echo ""
    echo "For adv_test, the newest train_<object>_*/Ep0_Texture_Noise.pt under"
    echo "PI_OBJECT_TEXTURE_ROOT is selected automatically."
}

if [ "$#" -lt 1 ]; then
    usage
    exit 2
fi

MODE=$1
shift

case "$MODE" in
    train|adv_test|clean_test|train_and_test) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac

OBJECT_TASKS='alphabet_soup 0
cream_cheese 1
salad_dressing 2
bbq_sauce 3
ketchup 4
tomato_sauce 5
butter 6
milk 7
chocolate_pudding 8
orange_juice 9'

latest_texture() {
    object_name=$1
    # Directory names contain sortable YYYYMMDD_HHMMSS timestamps, so lexical
    # order selects the newest completed training run for this object.
    find "$TEXTURE_ROOT" -maxdepth 2 -type f \
        -path "*/train_${object_name}_*/Ep0_Texture_Noise.pt" \
        -print 2>/dev/null | sort | tail -n 1
}

run_training() {
    echo "$OBJECT_TASKS" | while read -r object_name task_id; do
        [ -n "$object_name" ] || continue
        echo "============================================================"
        echo "[batch train] task=$task_id object=$object_name"
        "$RUN_ONE" \
            --run_mode train \
            --object_name "$object_name" \
            "$@"
    done
}

run_clean_test() {
    echo "$OBJECT_TASKS" | while read -r object_name task_id; do
        [ -n "$object_name" ] || continue
        echo "============================================================"
        echo "[batch clean_test] task=$task_id object=$object_name"
        "$RUN_ONE" \
            --run_mode clean_test \
            --object_name "$object_name" \
            --eval_task_id "$task_id" \
            "$@"
    done
}

run_adv_test() {
    echo "$OBJECT_TASKS" | while read -r object_name task_id; do
        [ -n "$object_name" ] || continue
        texture_path=$(latest_texture "$object_name")
        if [ -z "$texture_path" ]; then
            echo "No trained texture found for $object_name under $TEXTURE_ROOT" >&2
            echo "Run '$0 train ...' first, or set PI_OBJECT_TEXTURE_ROOT." >&2
            exit 1
        fi
        echo "============================================================"
        echo "[batch adv_test] task=$task_id object=$object_name"
        echo "[batch adv_test] texture=$texture_path"
        "$RUN_ONE" \
            --run_mode adv_test \
            --object_name "$object_name" \
            --eval_task_id "$task_id" \
            --load_texture_path "$texture_path" \
            "$@"
    done
}

case "$MODE" in
    train)
        run_training "$@"
        ;;
    adv_test)
        run_adv_test "$@"
        ;;
    clean_test)
        run_clean_test "$@"
        ;;
    train_and_test)
        run_training "$@"
        run_adv_test "$@"
        ;;
esac

echo "============================================================"
echo "[batch] all LIBERO-Object tasks completed: mode=$MODE"
