#!/bin/bash
# ckpt_guard.sh <ckpt_dir>: verify model.safetensors against sibling model.sha256 record (create if absent).
# Usage in chains before collect/train/eval: bash ckpt_guard.sh $CKPT || exit 1
# Rationale: 2026-10-16 a 49-byte append to a complete file corrupted it post-verification;
# this guard fails fast at chain start instead of mid-run.
set -u
D="${1:?ckpt dir}"
F=$D/model.safetensors
R=$D/model.sha256
[ -f "$F" ] || { echo "[ckpt_guard] missing $F"; exit 1; }
if [ ! -f "$R" ]; then
  sha256sum "$F" > "$R"
  echo "[ckpt_guard] recorded $(cut -d" " -f1 "$R")"
  exit 0
fi
got=$(sha256sum "$F" | cut -d" " -f1)
want=$(cut -d" " -f1 "$R")
if [ "$got" != "$want" ]; then
  echo "[ckpt_guard] HASH MISMATCH want=$want got=$got — 文件可能被追加/损坏，停止"
  exit 1
fi
echo "[ckpt_guard] hash OK $got"
