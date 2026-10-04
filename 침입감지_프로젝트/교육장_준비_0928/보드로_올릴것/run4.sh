#!/bin/bash
# 사용: bash run4.sh <이름> <s3|s4|lp> <소스: 0|파일> [옵션...]   (~/work/baby/anseong 에서)
cd ~/work/baby/anseong; source ~/work/env/bin/activate
name=$1; kind=$2; src=$3; shift 3
case $kind in s3) base=detect_baby_3state.py;; s4) base=detect_baby_4state.py;; *) base=detect_baby_lp.py;; esac
if [ "$src" = "0" ]; then prog=$base; else prog=${kind}_$(basename "$src" .mp4).py; sed "s/^CAM_SOURCE = 0 /CAM_SOURCE = \"$src\" /; s/cv2.VideoCapture(0)$/cv2.VideoCapture(\"$src\")/" $base > $prog; fi
F='모드\]|평균 FPS|실제 추론|건너뛴 프레임|평균          :|CPU 평균|CPU 온도|^    대기  |^    감시  |^    경계  |^    경보  |^    전체  |전환 횟수|클럭 전환|추론 [0-9]+회|추론 1회당|카메라 재개방|Traceback|Error|line [0-9]'
echo; echo "##### $name  [$prog $*]  max=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq) $(vcgencmd measure_temp)"
python $prog --no-gui --seconds 45 --power-csv "$name" "$@" 2>&1 | grep -E "$F" | head -34
sleep 8
