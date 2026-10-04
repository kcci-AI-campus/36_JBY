#!/bin/bash
# 저전력 실험 행렬 (라즈베리파이 5, ~/work/baby/anseong 에서 실행)
#   사용: bash lp_matrix.sh ABC     (돌릴 묶음 글자를 붙여 쓴다. 전체 = ABCDE)
#   detect_baby_lp.py = 3상태 판 + --threads/--alert-fps/--cam-off-standby/--idle-w + 단계별 에너지 추정
#   영상 입력용 사본은 sed 로 CAM_SOURCE 만 바꿔 만든다. 결과: 화면 출력 + lp_<이름>_summary.csv / _log.csv
set -u
PART=${1:-ABCDE}
cd ~/work/baby/anseong
source ~/work/env/bin/activate
F='모드\]|평균 FPS|실제 추론|건너뛴 프레임|평균          :|CPU 평균|CPU 온도|^    대기  |^    감시  |^    경보  |^    전체  |전환 횟수|단계별|1회 ms|inference |camera |preproc |postproc |judge |gate |쉼/대기|추론 1회당|유휴 |카메라 재개방|Traceback|Error'
POL=/sys/devices/system/cpu/cpufreq/policy0
MAXF=$(cat $POL/cpuinfo_max_freq)
run() {  # run <이름> <소스: 0|파일> <옵션들...>
  local name=$1; local src=$2; shift 2
  if [ "$src" = "0" ]; then prog=detect_baby_lp.py; else prog=lp_$(basename "$src" .mp4).py; sed "s/^CAM_SOURCE = 0 /CAM_SOURCE = \"$src\" /" detect_baby_lp.py > $prog; fi
  echo; echo "##### $name  [$prog $*]  freq_max=$(cat $POL/scaling_max_freq) $(vcgencmd measure_temp)"
  python $prog --no-gui --seconds 45 --power-csv "lp_$name" "$@" 2>&1 | grep -E "$F" | head -40
  sleep 8
}
setfreq() { echo $1 | sudo -n tee $POL/scaling_max_freq >/dev/null && echo "[cpufreq] max -> $(cat $POL/scaling_max_freq)"; }

if [[ $PART == *A* ]]; then
echo "===== A. INT8 vs FP32 (같은 320 입력, 경보 장면 = 아기+칼 계속) ====="
run A1_fp32_cap30  active60.mp4 --active-model ../v2_best.tflite --alert-fps 30
run A2_int8_cap30  active60.mp4 --active-model ../v2_int8.tflite --alert-fps 30
run A3_fp32_cap10  active60.mp4 --active-model ../v2_best.tflite --alert-fps 10
run A4_int8_cap10  active60.mp4 --active-model ../v2_int8.tflite --alert-fps 10
fi
if [[ $PART == *B* ]]; then
echo "===== B. CPU 클럭 상한 (DVFS) ====="
setfreq 1500000
run B1_fp32_cap30_1500MHz  active60.mp4 --active-model ../v2_best.tflite --alert-fps 30
run B2_int8_cap10_1500MHz  active60.mp4 --active-model ../v2_int8.tflite --alert-fps 10
setfreq $MAXF
fi
if [[ $PART == *C* ]]; then
echo "===== C. 스레드 수 ====="
run C1_fp32_cap30_2thr  active60.mp4 --active-model ../v2_best.tflite --alert-fps 30 --threads 2
fi
if [[ $PART == *D* ]]; then
echo "===== D. 감시 상태 상한 FPS (아기 움직임, 위험 없음) ====="
run D1_pan_watch6   pan60.mp4 --watch-fps 6
run D2_pan_watch15  pan60.mp4 --watch-fps 15
fi
if [[ $PART == *E* ]]; then
echo "===== E. 대기 상태에서 카메라 놓기 (실제 카메라, 빈 방) ====="
run E1_cam_keep  0
run E2_cam_off   0 --cam-off-standby
fi
if [[ $PART == *F* ]]; then
echo "===== F. 클럭 상한 1500 MHz 를 감시·대기 장면에도 ====="
setfreq 1500000
run F1_pan_watch10_1500MHz  pan60.mp4 --watch-fps 10
run F2_cam_standby_1500MHz  0
setfreq $MAXF
fi
if [[ $PART == *G* ]]; then
echo "===== G. 클럭 단계별 (FP32 320, 경보 cap30) ====="
echo "사용 가능 주파수: $(awk '{print $1}' /sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state 2>/dev/null | tr '
' ' ')"
for f in 1800000 2100000; do setfreq $f; run G_fp32_cap30_${f} active60.mp4 --active-model ../v2_best.tflite --alert-fps 30; done
setfreq $MAXF
fi
echo "===== 끝 ($PART) ====="; ls lp_*_summary.csv 2>/dev/null | tr '\n' ' '; echo
