#!/usr/bin/env bash
# Cho luot train m1 (pid 143606) xong roi chay doi chung cls_pw day du (62_clspw_control.py).
cd /home/xuand/vn_traffic_ai
while kill -0 143606 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] m1 xong, bat dau doi chung"
# giu luot doi chung cu (3/10 epoch) vi bai bao dang trich so lieu cua no
if [ -d runs/detect/ctrl_cw0_s0 ] && [ ! -d runs/detect/ctrl_cw0_s0_partial ]; then
  mv runs/detect/ctrl_cw0_s0 runs/detect/ctrl_cw0_s0_partial
fi
/home/xuand/miniconda3/envs/vehicle_env/bin/python scripts/verify/62_clspw_control.py
echo "[$(date '+%F %T')] doi chung xong, exit $?"
