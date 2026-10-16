#!/bin/bash
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
STAGE=$D/code/tex3d-deps/cloth_variants
TA=$EX/train_20261016_051845_p3e_eot/Universal_Cloth.png
LONG=$D/model/RLinf-Pi0-LIBERO-Long-SFT
cp $STAGE/30x20_solid/tablecloth.obj $LIVE/tablecloth.obj
cp $STAGE/30x20_solid/cloth_base.png $LIVE/cloth_base.png
sync
echo "[chain-P3c] assets 30x20_solid, rerun libero_10 with Long-SFT ckpt" >> $EX/watchdog_events.log
setsid nohup bash $PI/run_cloth.sh 0 eval --condition plain --suite libero_10 --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --ckpt $LONG > $EX/p3b_logs/p3c_plain_libero_10.log 2>&1 < /dev/null &
setsid nohup bash $PI/run_cloth.sh 1 eval --condition adv --suite libero_10 --tasks 0,1,2,3,4,5,6,7,8,9 --trials 10 --texture $TA --ckpt $LONG > $EX/p3b_logs/p3c_adv_libero_10.log 2>&1 < /dev/null &
