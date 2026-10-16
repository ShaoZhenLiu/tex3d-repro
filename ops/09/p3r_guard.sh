#!/bin/bash
D=/data/shaozhen.liu
PI=$D/code/tex3d/pi
EX=$D/code/tex3d/experiments/pi0_cloth
LIVE=$D/code/tex3d-deps/libero-pi/libero/libero/assets/tablecloth
STAGE=$D/code/tex3d-deps/cloth_variants
TA=$EX/train_20261016_051845_p3e_eot/Universal_Cloth.png
TB=$EX/train_20261016_051847_p3e_noeot/Universal_Cloth.png
while ! grep -aq "DONE-P3" $EX/watchdog_events.log; do sleep 60; done
sleep 30
cp $STAGE/30x20_solid/tablecloth.obj $LIVE/tablecloth.obj
cp $STAGE/30x20_solid/cloth_base.png $LIVE/cloth_base.png
sync
echo "[chain-P3r] assets switched back to 30x20_solid, rerunning both adv arms" >> $EX/watchdog_events.log
setsid nohup bash $PI/run_cloth.sh 1 eval --condition adv --suite libero_spatial --tasks 1,3,4,5,7,9 --trials 10 --texture $TA > $EX/p3_logs/p3_advA_libero_spatial.log 2>&1 < /dev/null &
setsid nohup bash $PI/run_cloth.sh 3 eval --condition adv --suite libero_spatial --tasks 1,3,4,5,7,9 --trials 10 --texture $TB > $EX/p3_logs/p3_advB_libero_spatial.log 2>&1 < /dev/null &
