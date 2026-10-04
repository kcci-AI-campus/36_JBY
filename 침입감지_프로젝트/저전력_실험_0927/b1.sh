#!/bin/bash
cd ~/work/baby/anseong; source ~/work/env/bin/activate
echo "===== 경계 시험 영상 만들기 ====="; python make_guard.py 2>&1 | grep -v INFO
bash h_idle.sh
echo "===== 4상태 판: 장면별 (320 FP32, 감시 5 FPS, 경계 15 FPS, 경보 전속력) ====="
bash run4.sh s4_babyonly s4 babyonly60.mp4
bash run4.sh s4_pan      s4 pan60.mp4
bash run4.sh s4_far      s4 far60.mp4
bash run4.sh s4_guard    s4 guard60.mp4
bash run4.sh s4_mix      s4 mix60.mp4
echo "===== b1 끝 ====="
