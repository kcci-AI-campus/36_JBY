#!/bin/bash
# 배치 2: 4상태 + 클럭 전환 + INT8 (조절 항목 전부), 3상태 비교, 카메라 놓기 재시험
cd ~/work/baby/anseong; source ~/work/env/bin/activate
echo "===== 4상태 + 상태별 클럭 전환 (대기·감시 1500 / 경계·경보 2400 MHz) ====="
bash run4.sh s4c_pan    s4 pan60.mp4    --clock-low 1500000 --clock-high 2400000
bash run4.sh s4c_guard  s4 guard60.mp4  --clock-low 1500000 --clock-high 2400000
bash run4.sh s4c_mix    s4 mix60.mp4    --clock-low 1500000 --clock-high 2400000
bash run4.sh s4c_active s4 active60.mp4 --clock-low 1500000 --clock-high 2400000 --alert-fps 30
echo "===== 조절 항목 전부: 4상태 + INT8 320 + 클럭 1500 고정 + 경보 30 FPS 상한 ====="
bash run4.sh s4all_pan    s4 pan60.mp4    --active-model ../v2_int8.tflite --clock-low 1500000 --clock-high 1500000
bash run4.sh s4all_mix    s4 mix60.mp4    --active-model ../v2_int8.tflite --clock-low 1500000 --clock-high 1500000 --alert-fps 30
bash run4.sh s4all_active s4 active60.mp4 --active-model ../v2_int8.tflite --clock-low 1500000 --clock-high 1500000 --alert-fps 30
echo "===== 3상태 판 비교 (같은 far/guard 영상) ====="
bash run4.sh s3_far   s3 far60.mp4
bash run4.sh s3_guard s3 guard60.mp4
echo "===== 카메라 놓기 재시험 (실제 카메라, 빈 방) ====="
bash run4.sh lp_E2b_cam_off lp 0 --cam-off-standby
echo "===== b2 끝 ====="
