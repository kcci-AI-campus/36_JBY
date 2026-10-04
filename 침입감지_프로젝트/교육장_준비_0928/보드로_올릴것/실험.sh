#!/bin/bash
# 9/28 교육장 실험 실행기 — 번호만 치면 실행하고, 끝나면 결과를 결과_0928.txt 에 한 줄로 자동 기록한다.
#
#   bash 실험.sh 목록            실험 번호와 설명, 걸리는 시간
#   bash 실험.sh B전체           장면 B 의 6가지 방식을 연달아 (장면은 처음 한 번만 만들면 됨, 약 7분)
#   bash 실험.sh B-3             하나만: 10초 준비 카운트다운 → 측정 → 자동 기록
#   bash 실험.sh B-3 0           준비 시간 0초로 바로 시작
#   bash 실험.sh 메모 "지연 3.1 2.8 3.4초"     손으로 잰 값을 결과 파일에 적기
#   bash 실험.sh 결과            지금까지 기록 보기
#
# 실험 설계 (분해 측정 + 조합 검증)
#   전력 = 유휴 + 초당 추론 횟수 × 추론 1회 에너지
#   - 구조(두 모드 / 3상태 / 4상태)는 '초당 추론 횟수' 를 정한다 → 장면마다 잰다 (-1 ~ -3)
#   - 정밀도·클럭은 '추론 1회 에너지' 를 정한다 → 4상태 위에서 하나씩 더해 잰다 (-4 ~ -6)
#   두 효과가 곱으로 따로 논다면 -3 과 -4~-6 의 비율이 장면과 상관없이 같게 나온다 = 검증.
cd ~/work/baby/anseong || { echo "~/work/baby/anseong 폴더가 없습니다. 업로드부터."; exit 1; }
source ~/work/env/bin/activate 2>/dev/null
R=~/work/baby/anseong/결과_0928.txt
PREP=${2:-10}

# 웹캠(C270) 자동노출의 '동적 프레임레이트' 를 끈다: 켜져 있으면 실내 조명에서 15 FPS 로 떨어져 측정 조건이 흔들린다 (9/28 새벽 확인, 끄면 30 FPS·밝기 동일)
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0 2>/dev/null && CAMFIX="카메라 30 FPS 고정" || CAMFIX="카메라 설정 못 바꿈(v4l2-ctl 없음)"
S4="python3 detect_baby_4state.py --no-gui --guard-gate --alert-fps 30"
TWO="python3 detect_baby.py --no-gui"
S3="python3 detect_baby_3state.py --no-gui"
RATE="python3 ../detect_rate.py --no-gui --wait 5 --frames 60"
M640=../v2e120_fp32_640.tflite
INT8="--active-model yolo11n_int8.tflite"
if sudo -n true 2>/dev/null; then
    CLKSW="--clock-low 1500000 --clock-high 2400000"     # 평소 1500, 경계·경보 2400
    CLK15="--clock-low 1500000 --clock-high 1500000"     # 모든 상태 1500
    CLK="클럭 전환 가능"
else
    CLKSW=""; CLK15=""
    CLK="주의: sudo 가 안 돼 클럭 전환은 빠짐 (-4, -6 은 -3 과 같고, -5 는 INT8 만)"
fi

declare -A DESC CMD NAME SEC
add() { DESC[$1]="$2"; NAME[$1]="$3"; SEC[$1]="$4"; CMD[$1]="$5"; }
scene() {   # $1=장면 글자  $2=장면 설명
    local S=$1 D=$2
    add $S-1 "$D | 두 모드 (팀원A 원본, 대기 SSDLite ↔ 고성능 YOLO11n FP32)"   ${S}_two      45 "$TWO --seconds 45 --power-csv ${S}_two"
    add $S-2 "$D | 3상태 (두 모드 + 감시)"                                       ${S}_s3       45 "$S3 --seconds 45 --power-csv ${S}_s3"
    add $S-3 "$D | 4상태 (FP32, 클럭 2400)"                                      ${S}_s4       45 "$S4 --seconds 45 --power-csv ${S}_s4"
    add $S-4 "$D | 4상태 + 상태별 클럭 (평소 1500, 경계·경보 2400, FP32)"        ${S}_s4clk    45 "$S4 --seconds 45 $CLKSW --power-csv ${S}_s4clk"
    add $S-5 "$D | 4상태 + INT8 + 상태별 클럭"                                   ${S}_s4i8clk  45 "$S4 --seconds 45 $INT8 $CLKSW --power-csv ${S}_s4i8clk"
    add $S-6 "$D | 4상태 + INT8 + 클럭 1500 고정 (전부)"                         ${S}_all      45 "$S4 --seconds 45 $INT8 $CLK15 --power-csv ${S}_all"
}
scene B "B 인형만 두고 가만히, 칼·콘센트는 화면 밖"
scene C "C 인형을 천천히 계속 움직임 (막대·끈으로, 몸은 화면 밖), 칼 없음"
scene D "D 인형을 칼 옆에 붙여 둠 (경보 계속)"
add 4-1  "빈 방, 교육장 배경만 5분 | 헛알림 세기"                             empty300 300 "$S4 --seconds 300 --power-csv empty300"
add 4-2  "인형만 가만히, 칼·콘센트 없음 5분 | 헛알림 세기"                   doll300  300 "$S4 --seconds 300 --power-csv doll300"
add 5-1  "소등 시험: 불 켬. 인형+칼 1.5~2 m | 640 FP32 검출률"               rate_5-1 15  "$RATE $M640 --conf 0.40 --out rate_5-1.csv"
add 5-2  "소등 시험: 불 끔 | 640 FP32 검출률"                                rate_5-2 15  "$RATE $M640 --conf 0.40 --out rate_5-2.csv"
add 5-3  "소등 시험: 휴대폰 손전등만 | 640 FP32 검출률"                       rate_5-3 15  "$RATE $M640 --conf 0.40 --out rate_5-3.csv"
add 6-1  "D 인형 칼 옆 | 4상태인데 경계·경보만 640 FP32 사용"                 D_hi640  45  "$S4 --seconds 45 --hi-model $M640 --power-csv D_hi640"
add 6-2  "D 인형 칼 옆 | 4상태, 최신 INT8 320, 문턱값 0.30"                   D_int8c30 45 "$S4 --seconds 45 $INT8 --conf 0.30 --power-csv D_int8c30"
add 6-3  "칼을 1.5 m 에, 인형도 같이 | 세 모델 검출률 (FP32 320 / INT8 320 / FP32 640)" rate_6-3 30 "$RATE yolo11n_fp32.tflite yolo11n_int8.tflite $M640 --conf 0.30,0.40 --out rate_6-3_150cm.csv"
add 6-4  "칼을 2 m 에, 인형도 같이 | 세 모델 검출률"                          rate_6-4 30 "$RATE yolo11n_fp32.tflite yolo11n_int8.tflite $M640 --conf 0.30,0.40 --out rate_6-4_200cm.csv"

order="B-1 B-2 B-3 B-4 B-5 B-6 C-1 C-2 C-3 C-4 C-5 C-6 D-1 D-2 D-3 D-4 D-5 D-6 4-1 4-2 5-1 5-2 5-3 6-1 6-2 6-3 6-4"

cooldown() {   # 다음 측정 전에 CPU 온도가 50 ℃ 아래로 내려오거나 90초가 될 때까지 쉰다 (발열이 뒤 측정에 섞이지 않게)
    local t
    for ((w=0; w<90; w+=5)); do
        t=$(vcgencmd measure_temp | grep -oE "[0-9]+\.[0-9]+")
        awk -v t="$t" 'BEGIN{exit !(t<50)}' && { echo "   식힘 끝: ${t} ℃"; return; }
        echo "   ■ 쉬는 중 (측정 안 함, 인형 그대로 둬도 됨)... ${t} ℃"; sleep 5
    done
}

record() {   # $1=이름 $2=번호 $3=설명
python3 - "$1" "$2" "$3" "$R" <<'PYEOF'
import sys, io, re, os, time
name, id_, desc, R = sys.argv[1:5]
line = None
if os.path.exists(name + "_summary.csv"):
    rows, meta = [], {}
    for ln in io.open(name + "_summary.csv", encoding="utf-8"):
        ln = ln.strip()
        if not ln: continue
        if ln.startswith("#,"):
            p = ln.split(",", 2); meta[p[1]] = p[2] if len(p) > 2 else ""; continue
        rows.append(ln.split(","))
    hdr = rows[0]; modes = []; total = "?"
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        if d.get("mode") == "total": total = d.get("avg_power_w", "?")
        elif d.get("avg_power_w"): modes.append("%s %s W (%s초)" % (d["mode_kr"], d["avg_power_w"], d["seconds"]))
    out = io.open(name + "_출력.txt", encoding="utf-8", errors="replace").read() if os.path.exists(name + "_출력.txt") else ""
    g = lambda p: (re.search(p, out).group(1) if re.search(p, out) else "?")
    trans = ", ".join("%s %s" % (k.replace("_count", ""), v) for k, v in meta.items() if k.endswith("_count") and k != "alarm_count")
    line = "%s | %s | 전체 평균 %s W | %s | 경보 %s회 | 전환 %s | CPU %s %% | 온도 %s C | FPS %s" % (
        id_, desc, total, " / ".join(modes), meta.get("alarm_count", "?"), trans,
        g(r"CPU 평균\s*:\s*([\d.]+)"), g(r"CPU 온도\s*:\s*([\d.]+)"), g(r"평균 FPS\s*:\s*([\d.]+)"))
else:
    out = io.open(name + "_출력.txt", encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(name + "_출력.txt") else []
    keep = [l.strip() for l in out if re.search(r"baby|knife|adult|outlet|검출률|tflite", l)]
    line = "%s | %s | %s" % (id_, desc, " ; ".join(keep[-12:]) if keep else "출력 파일 %s_출력.txt 참고" % name)
with io.open(R, "a", encoding="utf-8") as f:
    f.write(time.strftime("%H:%M") + " | " + line + "\n")
print("기록됨 → 결과_0928.txt:"); print("  " + line)
PYEOF
}

run_one() {   # $1=번호 $2=준비 초
    local id=$1 prep=$2
    if [ -z "${CMD[$id]}" ]; then echo "모르는 번호: $id   (bash 실험.sh 목록)"; return 1; fi
    local name=${NAME[$id]} sec=${SEC[$id]}
    pkill -f "detect_baby" 2>/dev/null && sleep 2     # 다른 감지 프로그램이 카메라를 잡고 있으면 끈다
    echo "========================================================"
    echo " $id  ${DESC[$id]}"
    echo " 약 ${sec}초 걸림.  $CLK · $CAMFIX"
    echo " 명령: ${CMD[$id]}"
    echo "========================================================"
    for ((i=prep; i>0; i--)); do echo "   ${PREPMSG:-장면을 만드세요}... $i"; sleep 1; done
    echo " ▶ 측정 시작 $(date +%H:%M:%S) — ${sec}초 동안 ${DURMSG:-장면을 그대로 두세요}"
    ${CMD[$id]} 2>&1 | tee "${name}_출력.txt" | grep -E "모드\]|평균 FPS|평균          :|CPU 평균|CPU 온도|경보 횟수|검출률|baby|knife|adult|outlet|Traceback|Error|line [0-9]"
    echo " ■ 측정 끝 $(date +%H:%M:%S) — 멈춰도 됩니다"
    record "$name" "$id" "${DESC[$id]}"
}

case "$1" in
  목록|list|"")
    echo "번호   초   장면 | 방식"
    for k in $order; do printf "%-5s %4s  %s\n" "$k" "${SEC[$k]}" "${DESC[$k]}"; done
    echo; echo "장면 하나를 통째로: bash 실험.sh B전체 / C전체 / D전체  (6가지 연달아, 사이사이 자동으로 식힘, 약 7분)"
    echo "경보→휴대폰 지연은 자동이 아님:  bash 실험.sh 지연   로 켜고 스톱워치로 잰 뒤  bash 실험.sh 메모 \"...\""
    echo "$CLK · $CAMFIX"; exit 0;;
  결과) cat "$R" 2>/dev/null || echo "아직 기록 없음"; exit 0;;
  메모) shift; echo "$(date +%H:%M) | 메모 | $*" >> "$R"; echo "기록됨: $(tail -1 "$R")"; exit 0;;
  지연)
    pkill -f "detect_baby" 2>/dev/null && sleep 2
    echo "경보 → 휴대폰 지연 재기. 알림·클립·웹을 켜고 돕니다. 인형을 칼 반경 안으로 밀어 넣고,"
    echo "보드 LED 가 빨강이 되는 순간 스톱워치 시작 → 휴대폰이 울리면 정지. 3회 (경보 사이 30초 이상 띄우기)."
    echo "첫 알림에서 휴대폰 캡처 2장(알림 화면, 탭해서 열린 실시간 화면). 끝나면 Ctrl+C."
    python3 detect_baby_4state.py --guard-gate --notify --clip --web --web-url auto --web-token 1806 --no-gui
    echo; echo "잰 값을 이렇게 적으세요:  bash 실험.sh 메모 \"지연 3.1 2.8 3.4초\""; exit 0;;
  B전체|C전체|D전체)
    S=${1:0:1}
    echo "장면 $S 6가지를 연달아 합니다. 첫 카운트다운 동안 장면을 만들고, 끝날 때까지(약 7분) 장면을 그대로 두세요."
    if [ "$S" = "C" ]; then
        DURMSG="인형을 계속 천천히 움직이세요 (좌우로 2초에 한 번 정도, 여섯 번 모두 같은 빠르기로)"
        PREPMSG="곧 측정 시작, 인형을 움직일 준비"
    fi
    for n in 1 2 3 4 5 6; do
        if [ $n -eq 1 ]; then run_one $S-$n $PREP
        else cooldown; if [ "$S" = "C" ]; then run_one $S-$n 5; else run_one $S-$n 0; fi; fi
        echo "   (${n}/6 완료)"
    done
    echo; echo "장면 $S 끝. 결과: bash 실험.sh 결과"; exit 0;;
esac

run_one "$1" "$PREP"
