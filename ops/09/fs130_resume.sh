#!/bin/bash
set -u
D=/data/shaozhen.liu/model/RLinf-Pi0-LIBERO-130-fullshot-SFT
B=https://hf-mirror.com/RLinf/RLinf-Pi0-LIBERO-130-fullshot-SFT/resolve/main/model.safetensors
for i in $(seq 1 200); do
  curl -sL --retry 3 -C - -o $D/model.safetensors "$B" && { echo "done attempt $i $(date)"; break; }
  echo "interrupt $i, resume in 15s $(date)"; sleep 15
done
