# -*- coding: utf-8 -*-
"""
위험구역 침입 감지 — 보드 추론 본체

강사님 예제 EX_03_Board_RPS_PreTrained_YOLO.py 의 구조를 그대로 쓰되,
이번 프로젝트에 맞게 고친 것. 바꾼 곳마다 [변경] 주석을 달았다.

클래스 : 0 baby   1 adult   2 knife   3 outlet

하는 일
    아기(baby)가 위험물(knife / outlet)에 가까워지면 경보한다.
    위험물 박스를 조금 부풀려 '위험 반경'으로 삼고, 아기 박스가 거기 닿으면 경보.
    반경을 고정 픽셀이 아니라 **위험물 박스 폭의 배수**로 잡아 원근을 자동 보정한다.

실행
    source ~/work/env/bin/activate
    cd ~/work/baby
    python3 detect_baby.py                      # 기본 모델(v2_best.tflite)
    python3 detect_baby.py best_int8.tflite     # 모델 지정
    python3 detect_baby.py --no-gui             # 화면 없이 (측정용)
    python3 detect_baby.py --trace              # 단계별 시간 측정 + 타임라인 기록
    python3 detect_baby.py --notify             # 경보 시 보호자 휴대폰으로 사진 전송
    python3 detect_baby.py --notify --clip      # 사진 + 경보 앞뒤 5초 영상까지
    python3 detect_baby.py --clip-raw           # 영상에 박스를 안 그림 (날것)
    python3 detect_baby.py --clip --clip-keep 100   # clips/ 에 최근 100개만 남김 (기본 50, 0 이면 무제한)
    python3 detect_baby.py --no-mirror          # 좌우 반전 끄기 (기본은 거울처럼 반전)
    python3 detect_baby.py --dwell-continuous   # 예전 체류 판정으로 (비교용)
    python3 detect_baby.py --dwell 0.8          # 체류 창을 늘린다 (느린 모델용)
    python3 detect_baby.py --dwell-ratio 0.4    # 위험 비율 문턱을 낮춘다
    python3 detect_baby.py --seconds 30         # 30초 돌리고 자동 종료 (측정용)

성능을 잴 때
    실제 기능을 다 켠 상태로 재야 한다. 추론 속도만 재는 bench_models.py 는
    웹 인코딩·그리기·판정을 하지 않아 실제보다 가볍게 나온다
    (FP32 기준 벤치 30.0 FPS / 실제 26.3 FPS).
        python3 detect_baby.py v2_best.tflite --web --seconds 30
        python3 detect_baby.py v2_int8.tflite --web --seconds 30
    끝에 나오는 "평균 FPS" 가 실사용 수치다.
    python3 detect_baby.py --web                # 브라우저로 실시간 보기 (포트 5000)
    python3 detect_baby.py --web --team         # 팀 보드 4대를 한 화면에 (한 명만)
    python3 detect_baby.py --web --notify --web-url https://xxxx.trycloudflare.com
                                                # 밖에서도 알림->실시간 화면이 열리게
    python3 detect_baby.py --web --web-token 암호    # 암호를 걸고
    python3 detect_baby.py --web --web-port 8080     # 포트를 바꿔서

웹으로 보기
    같은 네트워크 :  http://<보드IP>:5000
    밖에서       :  다른 터미널에서  cloudflared tunnel --url http://localhost:5000
    팀 페이지에 끼우기 :  <img src="http://<보드IP>:5000/video_feed">
    주소는 팀원 app.py 와 같다 (/video_feed, /api/status)

LED
    초록 = 대기 / 노랑 = 아기 감지 / 빨강 = 위험(경보)
    준비 : pip install gpiozero lgpio
    배선과 시험 : python3 led_test.py     (먼저 이걸로 확인하고 본 코드를 돌릴 것)
    RGB LED 한 개짜리 배선이면 이 파일의 LED_MODE 를 "rgb" 로 바꾼다

--trace 를 붙이면 (trace_util.py 가 같은 폴더에 있어야 한다)
    q 로 끝낼 때 한 프레임을 이루는 단계가 각각 몇 ms 인지 표로 나오고,
    trace_baby.json 이 만들어진다. PC 로 가져와 https://ui.perfetto.dev 에
    끌어다 놓으면 프레임 하나하나의 타임라인과 코어별 CPU 사용률이 보인다.

    재는 구간
        frame      한 프레임 전체
         camera     카메라에서 한 장 읽기
         preproc    RGB 변환 + letterbox
         normalize  0~1 나누기 / INT8 환산 / NCHW 전치
         inference  모델 실행 (invoke)
         postproc   신뢰도 필터 + NMS + 좌표 복원
         judge      아기-위험물 거리 판정
         led        LED 켜고 끄기
         draw       박스·원·글자 그리기
         imshow     화면에 넘기기
         waitKey    키 대기 (실제 화면 전송이 여기서 일어난다)
    그리고 경보가 울린 순간(ALARM)과 상태(state: danger/baby 개수)가 함께 찍힌다.

    쓰는 법 : 세 모델(best / best_int8 / best_w8a32)을 각각 30초씩
             --trace 로 돌리면, 양자화가 줄인 시간이 정확히 inference 구간에서만
             줄었는지 확인할 수 있다. 다른 구간이 병목이면 경량화는 효과가 없다.
"""
import os
import sys
import time

import numpy as np
import cv2

# 추론 엔진 — 있는 것을 순서대로 쓴다.
#   보드(라즈베리파이)   : ai_edge_litert  (pip install ai-edge-litert)
#   구버전 보드          : tflite_runtime
#   노트북(윈도우/맥)    : ai_edge_litert 이 안 깔리면 tensorflow 안의 lite 를 쓴다
#                          (pip install tensorflow)  무겁지만 동작은 같다
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite          # 구버전 보드 대비
    except ImportError:
        import tensorflow as _tf                             # 노트북 대비
        tflite = _tf.lite

# ─────────────────────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────────────────────
# 기본 모델 — 2026-09-22 실사용 검출률 측정(detect_rate.py)으로 정했다.
#
#   같은 60프레임으로 모델 5종을 비교한 결과 (눕힌 칼 + 인형 장면)
#       모델            baby   adult  knife   추론
#       by_best  (우리)  75%    63%     0%    26.1ms   <- 칼을 아예 못 찾음
#       by_w8a32 (우리)  75%    57%     0%    25.5ms
#       v2_int8  (팀원)  75%     0%    85%     9.1ms
#       v2_best  (팀원)  77%     3%    93%    24.9ms   <- 채택
#
#   우리 모델은 양자화 전(FP32)에도 knife 0% 였으므로 양자화 탓이 아니라
#   학습 데이터 탓이다(merged_v3 의 knife 박스가 153개로 가장 적었다).
#   adult 는 경보 판정에 쓰이지 않으므로(SUBJECT=0, DANGER=(2,3))
#   v2_best 의 adult 3% 는 기능 손실이 아니다.
#
#   대가 : v2_best 는 float32 라 CPU 가 약 50% 든다. v2_int8 은 약 18% 지만
#          knife 검출률이 93% -> 85%, 신뢰도가 0.81 -> 0.50 으로 떨어진다.
#          CONF_TH 를 낮추면 INT8 이 회복되는지 확인 중이다.
MODEL = "v2_best.tflite"            # 인자로 덮어쓸 수 있다
IMG_SIZE = 320
CONF_TH = 0.40
IOU_TH = 0.45
THREADS = 4                         # 추론 스레드. 상태 기계에서 바꿔 쓸 값
WAIT_MS = 1                         # cv2.waitKey 대기(ms). 10이면 화면이 발목을 잡는다

# 카메라 버퍼 크기 — 2026-09-23 측정으로 정했다.
#   1 로 두면 한 장을 읽어 처리하는 동안 도착한 다음 프레임이 버려진다.
#   실측 : 버퍼1 -> 16.8~20.0 FPS (간격 36~68ms 들쭉날쭉)
#          버퍼3 -> 30.0 FPS      (간격 32~36ms 일정)
#   읽기 스레드를 써도 버퍼가 1이면 15 FPS 였으므로 버퍼가 원인이다.
#   대신 프레임이 최대 3장까지 줄을 서므로 지연이 조금 늘 수 있다
#   (추론이 33ms 보다 빠르면 줄이 쌓이지 않아 실제로는 거의 없다).
CAM_BUFFER = 3


# 좌우 반전 — 거울처럼 보이게 한다.
#   반전하지 않으면 화면을 보며 인형을 움직일 때 방향이 반대로 느껴져 시연이 어렵다.
#   **카메라에서 읽은 직후에 뒤집으므로** 검출·판정·박스 좌표가 전부 같은 그림 기준이 된다.
#   (그린 뒤에 뒤집으면 글자까지 뒤집힌다)
#   --no-mirror 로 끌 수 있다. 비용은 프레임당 0.2ms 수준.
MIRROR = True

# [변경] 클래스 4개
NAMES = {0: "baby", 1: "adult", 2: "knife", 3: "outlet"}
COLORS = {0: (0, 200, 255),         # baby   - 주황
          1: (200, 200, 200),       # adult  - 회색
          2: (0, 0, 255),           # knife  - 빨강
          3: (0, 0, 255)}           # outlet - 빨강
DANGER = (2, 3)                     # 위험물 클래스
SUBJECT = 0                         # 감지 주체 = baby

# [추가] 판정
DANGER_SCALE = 1.5                  # 위험 반경 = 위험물 박스 폭 x 이 값
DWELL_SEC = 0.5                     # 이 시간만큼 살펴보고 경보를 낼지 정한다
COOLDOWN_SEC = 3.0                  # 경보 후 이 시간 동안은 다시 안 울린다

# 체류 판정 방식 — 2026-09-23 에 바꿨다.
#
#   "continuous" (예전) : DWELL_SEC 동안 **한 프레임도 빠짐없이** 위험이어야 경보.
#       한 장만 놓쳐도 타이머가 0 으로 돌아간다.
#       그래서 FPS 가 높을수록 불리하다 — 30 FPS 면 0.5초에 15장을 전부 맞혀야 한다.
#       검출률이 93% 라면 15장 연속 성공 확률은 0.93^15 = 34% 에 불과하다.
#
#   "ratio" (지금) : 최근 DWELL_SEC 안의 프레임 중 DWELL_RATIO 이상이 위험이면 경보.
#       한두 장 놓쳐도 경보가 유지되고, **FPS 가 높을수록 표본이 많아져 더 정확해진다.**
#
#   --dwell-continuous 를 주면 예전 방식으로 돌릴 수 있다(비교용).
#   어느 쪽으로 돌리든 종료할 때 두 방식의 경보 횟수를 나란히 보여 준다.
DWELL_MODE = "ratio"
DWELL_RATIO = 0.6                   # 최근 창의 60% 이상이 위험이면 경보

# [추가] LED (GPIO). 없으면 자동으로 건너뛴다
#   LED_MODE "separate" : LED 3개를 따로 단 배선 (초록·노랑·빨강)
#   LED_MODE "rgb"      : 다리 4개짜리 RGB LED 한 개 (노랑 = 빨강+초록)
#   배선과 시험은 led_test.py 참고.  핀 번호는 BCM (GPIO17 = 헤더 11번 핀)
# LED 3개를 따로 꽂는 경우가 "separate", 세 색이 한 알에 든 RGB LED 가 "rgb".
#
#   상태는 세 가지다.
#       초록  정상      아기가 화면에 없다
#       중간  감지      아기는 보이는데 위험물과 떨어져 있다
#       빨강  경보      위험 반경 안에 0.5초 이상 머물렀다
#
#   가진 LED 가 빨강/초록/파랑 3개라면 노랑을 만들 수 없다.
#   (빨강+초록으로 노랑이 되는 것은 세 색이 한 알에 들어 광학적으로 섞일 때뿐이고,
#    LED 가 따로 떨어져 있으면 그냥 두 개가 켜질 뿐이다.)
#   그래서 **가운데 자리를 파랑으로 쓴다.** 핀과 동작은 그대로다.
LED_MODE = "separate"
LED_PINS = {"green": 17, "mid": 27, "red": 22}          # separate 일 때 (가운데=파랑 권장)
LED_MID_NAME = "파랑"                                    # 로그에 찍을 이름. 노랑을 꽂았으면 "노랑"
RGB_PINS = {"red": 17, "green": 27, "blue": 22}         # rgb 일 때 (한 알짜리 RGB)
RGB_COMMON_ANODE = False        # 공통 다리를 3.3V 에 넣었으면 True
USE_LED = True

# [추가] 음성 경고 (wav 파일이 있으면 재생)
ALARM_WAV = "alarm.wav"
USE_SOUND = True

SHOW_GUI = "--no-gui" not in sys.argv
USE_TRACE = "--trace" in sys.argv       # Perfetto 타임라인 기록 (trace_util.py 필요)
USE_NOTIFY = "--notify" in sys.argv    # 보호자 휴대폰 알림 (notify.py + notify.json)
USE_CLIP = ("--clip" in sys.argv) or ("--clip-raw" in sys.argv)        # 경보 순간 앞뒤를 동영상으로 (clip.py)
CLIP_RAW = "--clip-raw" in sys.argv    # 영상에 박스를 안 그리고 날것으로
if "--no-mirror" in sys.argv:
    MIRROR = False
if "--dwell-continuous" in sys.argv:
    DWELL_MODE = "continuous"        # 예전 방식으로 (비교용)


def _argval(flag, default=""):
    """--flag 값  형태의 인자를 읽는다."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# ── 검출 문턱값을 명령줄에서 바꾼다 ────────────────────────
#   이 값이 '정확도' 를 좌우하는 거의 유일한 설정이다.
#   몇 % 이상을 "찾았다" 로 칠 것인가.
#
#     낮추면 (0.25)  멀리 있는 칼도 잡는다.  대신 책·옷도 칼로 잡는다(오검출)
#     올리면 (0.50)  헛것을 덜 잡는다.      대신 먼 칼을 놓친다
#
#   전처리(letterbox + /255)와 NMS 는 학습 때와 맞춰져 있어 건드릴 것이 없다.
#   즉 같은 모델이라면 코드에서 결과를 바꿀 수 있는 손잡이는 여기뿐이다.
#
#     python3 detect_baby.py <모델> --conf 0.25
CONF_TH = float(_argval("--conf", str(CONF_TH)))
IOU_TH = float(_argval("--iou", str(IOU_TH)))


# ── 움직임 게이팅 : 화면이 안 변하면 추론을 건너뛴다 ───────
#
#   원래 기획은 초음파 센서로 "뭔가 가까이 왔을 때만 추론" 하는 것이었다.
#   센서를 쓸 수 없게 되어, 같은 원리를 소프트웨어로 구현한다.
#
#     초음파 : 반사 거리로 '뭔가 있는지' 를 판단  -> 있을 때만 추론
#     차분   : 직전 프레임과의 픽셀 차이로 판단   -> 변했을 때만 추론
#
#   비용 비교 (FP32 @640 기준, 실측)
#     추론 한 번        약 129 ms
#     차분 한 번        약  1 ms       <- 130배 싸다
#
#   아기 방은 대부분의 시간이 정적이다. 아무도 없는 동안 129 ms 짜리 추론을
#   계속 돌리는 것은 낭비다.
#
#   안전 장치 세 가지 (감시 공백을 만들지 않기 위해)
#     1) 지금 위험 상태면 절대 건너뛰지 않는다
#     2) 연속으로 GATE_MAX_SKIP 장을 건너뛰면 강제로 한 번 추론한다
#        (아주 천천히 움직여 차분에 안 걸리는 경우 대비)
#     3) 건너뛴 프레임은 직전 검출 결과를 그대로 쓴다. 비워 두지 않는다
#
#     python3 detect_baby.py <모델> --gate
GATE = "--gate" in sys.argv
GATE_TH = float(_argval("--gate-th", "0.6"))   # 변한 픽셀 비율(%) 이 이 값 미만이면 정지로 본다
GATE_MAX_SKIP = int(_argval("--gate-max", "15"))   # 연속으로 건너뛸 수 있는 최대 장수
GATE_SIZE = 160                                # 차분은 작게 줄여서 본다(빠르다)

# 게이팅 2단계 : 추론을 건너뛰는 것만으로는 전력이 덜 준다.
#
#   1단계(건너뛰기)만 켰을 때의 실측 :
#       게이팅 끔   6.03 W   12.4 FPS   CPU 91%
#       게이팅 켬   3.04 W   27.9 FPS   CPU 22%
#
#   전력이 줄긴 했지만 **FPS 가 12 -> 28 로 올라갔다.**
#   추론을 안 하니 루프가 그만큼 빨리 돌아버린 것이다.
#   아무 일도 없는데 초당 28장씩 카메라를 읽고 차분을 돌리는 것은 낭비다.
#
#   그래서 조용할 때는 **일부러 쉰다.** 원래 초음파 설계가 하려던 것과 같다 —
#   센서가 반응할 때까지 보드는 자고 있는 것.
#
#   움직임이 감지되면 그 즉시 전속력으로 돌아온다(쉬는 것은 건너뛰는 중일 때만).
#   조용할 때 8 FPS 면 움직임이 시작되고 0.13초 안에 알아챈다.
#   경보 판정이 0.5초를 보므로 여유가 있다.
GATE_IDLE_FPS = float(_argval("--gate-idle-fps", "8"))   # 0 이면 안 쉰다


USE_WEB = "--web" in sys.argv          # 브라우저로 실시간 보기 (web.py + flask)
USE_TEAM = "--team" in sys.argv        # 팀 보드 여러 대를 한 화면에 (--web 과 같이)
WEB_PORT = int(_argval("--web-port", "5000"))
WEB_TOKEN = _argval("--web-token", "")   # 주면 ?k=토큰 이 있어야 열린다

# 알림을 눌렀을 때 열릴 주소를 직접 지정한다.
#   기본값은 보드의 랜 IP(예: http://10.10.15.61:5000/) 인데, 그 주소는
#   **같은 네트워크 안에서만** 통한다. 밖에서도 실시간 화면을 보려면
#   터널로 받은 주소를 여기에 넣어야 한다.
#       cloudflared tunnel --url http://localhost:5000     (다른 터미널에서, 켜 둔 채로)
#       python3 detect_baby.py ... --web --notify --web-url https://xxxx.trycloudflare.com
#
#   주소는 터널을 껐다 켤 때마다 바뀐다. 매번 복사해 붙이는 것은 실수하기 쉽다
#   (옛 주소가 실린 알림을 눌러 "접속이 안 된다" 고 헤맨 적이 있다).
#   그래서 --web-url auto 를 주면 cloudflared 에게 직접 물어본다.
#       python3 detect_baby.py ... --web --notify --web-url auto
WEB_URL_OVERRIDE = _argval("--web-url", "")


def ask_cloudflared(timeout=1.0):
    """켜져 있는 cloudflared 에게 지금 주소가 뭔지 물어본다.

    cloudflared 는 자기 상태를 알려 주는 작은 서버를 127.0.0.1 에 띄운다
    (포트는 20241 부터 비어 있는 것 하나). 그 중 /quicktunnel 이
    {"hostname": "vocals-thunder-....trycloudflare.com"} 을 돌려준다.
    보드 안에서만 열리는 주소라 밖으로 새지 않는다.
    """
    try:
        import requests
    except Exception:
        return ""
    for port in range(20241, 20246):
        try:
            r = requests.get("http://127.0.0.1:%d/quicktunnel" % port, timeout=timeout)
            if r.status_code != 200:
                continue
            host = (r.json() or {}).get("hostname", "")
            if host:
                return "https://" + host
        except Exception:
            continue
    return ""

# 체류 판정 값을 명령줄에서 조정한다. 시연 중 바로 바꿔 보려고 뺐다.
#   FPS 가 낮으면 0.5초 창에 프레임이 몇 장 안 들어와 판정이 거칠어진다.
#       30 FPS -> 15장,  21 FPS -> 11장,  7.5 FPS -> 4장
#   4장이면 60% 문턱이 "3장 중 2장" 수준이라 한 장 차이로 널뛴다.
#   느린 모델을 쓸 때는 DWELL_SEC 을 늘리거나 RATIO 를 낮춘다.
DWELL_SEC = float(_argval("--dwell", str(DWELL_SEC)))
DWELL_RATIO = float(_argval("--dwell-ratio", str(DWELL_RATIO)))

# 정해진 시간 뒤 자동 종료. 모델별 실사용 FPS 를 같은 조건으로 재는 데 쓴다.
#   성능 수치는 **실제 기능을 다 켠 상태**로 잰 값을 기준으로 삼아야 한다.
#   추론 속도만 재는 bench_models.py 는 웹 인코딩·그리기·판정을 하지 않아
#   실제보다 가볍게 나온다 (FP32 기준 벤치 30.0 FPS / 실제 26.3 FPS).
RUN_SEC = float(_argval("--seconds", "0"))   # 0 이면 수동 종료

# ─────────────────────────────────────────────────────────────
#  팀 보드 목록 — --team 으로 띄울 때 한 화면에 모을 보드들
#    각 보드에서 이 프로그램이 --web 으로 돌고 있어야 한다(포트 5000).
#    꺼져 있는 보드는 '연결 없음' 으로만 표시되고 나머지는 멀쩡하다.
#    밖에서 볼 때는 주소를 각자의 https 터널 주소로 바꾼다(http 와 섞으면 차단된다).
# ─────────────────────────────────────────────────────────────
TEAM_BOARDS = [
    ("정보윤", "http://10.10.15.61:5000"),
    ("팀원A", "http://10.10.15.62:5000"),
    ("팀원B", "http://10.10.15.63:5000"),
    ("팀원C", "http://10.10.15.64:5000"),
]
# --web-port 5000 처럼 '옵션의 값'은 모델 이름이 아니므로 빼고 고른다
_skip = set()
for _f in ("--web-port", "--web-token", "--seconds", "--web-url"):
    if _f in sys.argv:
        _i = sys.argv.index(_f)
        if _i + 1 < len(sys.argv):
            _skip.add(_i + 1)
args = [a for i, a in enumerate(sys.argv)
        if i >= 1 and not a.startswith("--") and i not in _skip]
if args:
    MODEL = args[0]

# ─────────────────────────────────────────────────────────────
#  트레이싱 (선택)
#    --trace 를 주면 단계별 시간과 코어별 CPU 를 trace_baby.json 으로 남긴다.
#    보는 법 : PC 로 파일을 가져와 https://ui.perfetto.dev 에 끌어다 놓기
# ─────────────────────────────────────────────────────────────
tr = None
if USE_TRACE:
    for cand in (".", os.path.expanduser("~/work/examples")):
        if os.path.exists(os.path.join(cand, "trace_util.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            break
    try:
        from trace_util import Tracer
        tr = Tracer("trace_baby.json")
        print("[trace] 기록 시작 -> trace_baby.json")
    except Exception as e:
        print("[trace] 사용 안 함 (%s)" % e)
        print("        trace_util.py 를 이 폴더에 두거나 ~/work/examples/ 에 두세요")


class _Null:
    """트레이싱을 끈 상태에서도 같은 코드가 돌아가게 하는 빈 껍데기"""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def span(name, tid=1):
    return tr.span(name, tid) if tr else _Null()


# ─────────────────────────────────────────────────────────────
#  보호자 휴대폰 알림 (선택)
#    --notify 를 주면 경보가 울릴 때 그 순간의 화면을 사진으로 함께 보낸다.
#    보내는 일은 notify.py 가 별도 스레드에서 하므로 영상은 멈추지 않는다.
#    설정 : python3 notify.py --init  ->  휴대폰에 앱 설치  ->  --test 로 확인
# ─────────────────────────────────────────────────────────────
noti = None
if USE_NOTIFY:
    try:
        from notify import Notifier
        noti = Notifier()
        if not noti.enabled:
            noti = None
    except Exception as e:
        print("[알림] 사용 안 함 (%s)" % e)
        print("       python3 notify.py --init 으로 설정을 만드세요")


# ─────────────────────────────────────────────────────────────
#  경보 순간 영상 (선택)
#    --clip 을 주면 경보 앞 3초 + 뒤 2초를 동영상으로 묶어 보낸다.
#    사진은 "이미 가까이 갔다"만 보여 주지만, 영상은 "어떻게 접근했는가"가 보인다.
#    --notify 와 함께 써야 전송된다 (단독으로 쓰면 파일로만 남는다).
# ─────────────────────────────────────────────────────────────
WEB_URL = None      # 웹을 켠 뒤에 채운다 (아래 웹 블록 참고)

clipper = None
if USE_CLIP:
    try:
        from clip import ClipRecorder
        clipper = ClipRecorder(noti, pre_sec=3.0, post_sec=2.0, fps=20.0,
                               draw=not CLIP_RAW, names=NAMES, colors=COLORS,
                               min_gap_sec=30.0, click=WEB_URL,
                               max_files=int(_argval("--clip-keep", "50")))
        if noti is None:
            print("[클립] --notify 가 없어 전송은 하지 않고 clips/ 폴더에만 저장합니다")
    except Exception as e:
        print("[클립] 사용 안 함 (%s)" % e)


# ─────────────────────────────────────────────────────────────
#  브라우저로 실시간 보기 (선택)
#    --web 을 주면 같은 네트워크의 PC·휴대폰에서 http://<보드IP>:5000 으로 볼 수 있다.
#    밖에서 보려면 따로 터널을 연다 :  cloudflared tunnel --url http://localhost:5000
#
#    팀원의 app.py 와 주소를 같게 맞췄다(/video_feed, /api/status).
#    다만 추론 루프는 여기(본 파일)에서만 돌고 웹은 결과를 구경만 한다.
#    app.py 처럼 요청 안에서 추론하면, 아무도 안 볼 때 감시가 멈추고
#    두 명이 보면 같은 interpreter 에 동시에 invoke 가 들어간다.
# ─────────────────────────────────────────────────────────────
web = None
if USE_WEB:
    try:
        from web import WebServer
        web = WebServer(port=WEB_PORT, token=WEB_TOKEN,
                        model_name=os.path.basename(MODEL),
                        boards=TEAM_BOARDS if USE_TEAM else None)
        if web.enabled:
            web.start()
        else:
            web = None
    except Exception as e:
        print("[웹] 사용 안 함 (%s)" % e)
        print("     쓰려면 : pip install flask")

# 알림을 눌렀을 때 열릴 주소.
#   웹이 켜져 있으면 휴대폰에서 알림을 탭하는 것만으로 실시간 화면으로 넘어간다.
#   (웹 서버가 만들어진 뒤라야 주소를 알 수 있어서 여기서 채운다)
if web:
    try:
        _ov = WEB_URL_OVERRIDE
        if _ov.lower() == "auto":
            _ov = ask_cloudflared()
            if _ov:
                print("[알림] cloudflared 에서 주소를 받아왔습니다 :", _ov)
            else:
                print("[알림] cloudflared 를 못 찾았습니다. 터널이 켜져 있는지 보세요 :")
                print("       cloudflared tunnel --url http://localhost:%d" % WEB_PORT)
                print("       일단 랜 IP 주소로 알림을 보냅니다(집 안에서만 열립니다).")
        if _ov:
            WEB_URL = _ov.rstrip("/") + "/"
            if WEB_TOKEN:
                WEB_URL += "?k=" + WEB_TOKEN
            print("[알림] 알림을 누르면 열릴 주소 :", WEB_URL)
        else:
            _ip = web.my_ip()
            if _ip:
                WEB_URL = "http://%s:%d/%s" % (_ip, WEB_PORT,
                                               ("?k=" + WEB_TOKEN) if WEB_TOKEN else "")
                print("[알림] 알림을 누르면 열릴 주소 :", WEB_URL)
                print("       이 주소는 같은 네트워크에서만 열립니다.")
                print("       밖에서도 보려면 터널 주소를 --web-url 로 주세요.")
        if clipper:
            clipper.click = WEB_URL

        # ── 터널이 정말 우리에게 연결되는지 확인한다 ──────
        #   cloudflared 는 '어느 포트로 보낼지' 를 실행할 때 사람이 친다.
        #   여기서 오타가 나면(예: 5000 을 500 으로) 주소는 멀쩡히 생기지만
        #   아무것도 안 열린다. Cloudflare 로그에만 영어로 오류가 찍혀서
        #   원인을 찾기 어렵다. 그래서 시작할 때 한 번 실제로 접속해 본다.
        if WEB_URL_OVERRIDE.lower() == "auto" and WEB_URL:
            def _check_tunnel(url):
                import threading
                def run():
                    try:
                        import requests
                    except Exception:
                        return
                    # 터널이 열리는 데 몇 초 걸린다. 몇 번 다시 해 본다.
                    for _ in range(6):
                        time.sleep(3)
                        try:
                            r = requests.get(url, timeout=6)
                        except Exception:
                            continue
                        if r.status_code < 400:
                            print("[터널] 확인됨 : 바깥에서 이 보드까지 연결됩니다")
                            return
                        if r.status_code == 403:
                            print("[터널] 연결은 되는데 암호가 틀렸습니다(--web-token 확인)")
                            return
                    print("[터널] !! 바깥에서 이 보드에 닿지 못합니다.")
                    print("       cloudflared 를 띄운 터미널의 포트를 확인하세요.")
                    print("       맞는 명령 :  cloudflared tunnel --url http://localhost:%d"
                          % WEB_PORT)
                    print("       (0 을 하나 빠뜨려 500 으로 치는 실수가 잦습니다)")
                threading.Thread(target=run, daemon=True).start()
            _check_tunnel(WEB_URL)

        # ── 시작할 때 주소를 휴대폰으로 보낸다 ────────────
        #   무료 터널은 껐다 켤 때마다 주소가 바뀐다.
        #   그걸 사람이 받아적어 휴대폰에 치는 과정에서 계속 실수가 났다
        #   (죽은 옛 주소를 열고 Error 1033 을 보는 일이 반복됐다).
        #   그래서 프로그램이 직접 보낸다. 탭하면 바로 열린다.
        if noti and WEB_URL:
            try:
                noti.send("감시를 시작했습니다. 눌러서 실시간 화면을 여세요.",
                          title="👶 감시 시작", click=WEB_URL)
                # 알림에는 '최소 30초 간격' 제한이 있다. 시작 알림이 그 시간을
                # 써 버리면, 시작하자마자 일어난 진짜 경보가 막힌다.
                # 시작 알림은 안내일 뿐이므로 간격 계산에서 빼 둔다.
                noti._last = 0
                print("[알림] 시작 알림으로 이 주소를 휴대폰에 보냈습니다.")
                print("       알림을 누르면 바로 열립니다. 주소를 손으로 칠 필요가 없습니다.")
            except Exception as e:
                print("[알림] 시작 알림을 보내지 못했습니다 :", e)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
#  LED
# ─────────────────────────────────────────────────────────────
leds = {}
_pins = RGB_PINS if LED_MODE == "rgb" else LED_PINS
if USE_LED:
    try:
        from gpiozero import LED
        leds = {k: LED(v, active_high=not (LED_MODE == "rgb" and RGB_COMMON_ANODE))
                for k, v in _pins.items()}
        # 핀을 열었다는 뜻이지 LED 가 실제로 꽂혀 있다는 뜻이 아니다.
        # GPIO 는 아무것도 연결되지 않아도 열린다.
        print("[LED] GPIO 핀 열림 (%s) : %s" % (LED_MODE, _pins))
        _mid = "노랑(빨강+초록)" if LED_MODE == "rgb" else LED_MID_NAME
        print("      초록 = 정상(아기 없음) / %s = 아기 감지 / 빨강 = 경보" % _mid)
        print("      LED 가 실제로 켜지는지는 python3 led_test.py 로 확인하세요")
    except Exception as e:
        print("[LED] 사용 안 함 (%s)" % e)
        print("      쓰려면 : pip install gpiozero lgpio")
        print("      배선 확인 : python3 led_test.py")
        leds = {}

# 상태 이름 -> 켤 다리 목록.
#   RGB LED 는 다리가 빨강/초록/파랑뿐이라 노랑을 빨강+초록으로 만든다.
if LED_MODE == "rgb":
    # 한 알짜리 RGB 는 빨강+초록이 섞여 노랑이 된다.
    LED_STATE = {"green": ["green"], "mid": ["red", "green"], "red": ["red"]}
else:
    LED_STATE = {"green": ["green"], "mid": ["mid"], "red": ["red"]}


def set_led(color):
    """상태에 맞는 색을 켜고 나머지는 끈다."""
    if not leds:
        return
    want = LED_STATE.get(color, [])
    for k, d in leds.items():
        d.on() if k in want else d.off()


# ─────────────────────────────────────────────────────────────
#  소리
# ─────────────────────────────────────────────────────────────
def play_alarm():
    if not USE_SOUND or not os.path.exists(ALARM_WAV):
        return
    os.system("aplay -q %s &" % ALARM_WAV)     # 재생이 끝날 때까지 기다리지 않는다


# ─────────────────────────────────────────────────────────────
#  모델
# ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL):
    print("모델 파일이 없습니다:", MODEL)
    sys.exit(1)

try:
    interpreter = tflite.Interpreter(model_path=MODEL, num_threads=THREADS)
except TypeError:                     # num_threads 미지원 버전
    interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
in_idx, out_idx = inp["index"], out["index"]
in_shape = list(inp["shape"])

# [변경] 입력이 NCHW 인지 NHWC 인지 자동으로 판별한다.
#   강사님 모델은 (1,3,320,320) NCHW 였지만,
#   Ultralytics 로 직접 내보낸 tflite 는 보통 (1,320,320,3) NHWC 다.
#   여기서 틀리면 오류가 나거나 — 더 나쁘게는 엉뚱한 결과가 조용히 나온다.
NCHW = (in_shape[1] == 3)
IMG_SIZE = in_shape[2] if NCHW else in_shape[1]

# [추가] INT8 모델 대응. 입출력이 정수면 scale/zero_point 로 환산해야 한다.
in_q = inp.get("quantization", (0.0, 0))
out_q = out.get("quantization", (0.0, 0))
IN_INT = inp["dtype"] in (np.int8, np.uint8)
OUT_INT = out["dtype"] in (np.int8, np.uint8)

print("모델      :", MODEL)
print("입력      :", in_shape, "->", "NCHW" if NCHW else "NHWC",
      "dtype", np.dtype(inp["dtype"]).name, ("양자화 %s" % (in_q,)) if IN_INT else "")
print("출력      :", list(out["shape"]), "dtype", np.dtype(out["dtype"]).name,
      ("양자화 %s" % (out_q,)) if OUT_INT else "")
print("입력 크기 :", IMG_SIZE)
print("문턱값    : conf %.2f / iou %.2f   (--conf 로 바꿈)" % (CONF_TH, IOU_TH))
print("체류 판정 : 최근 %.2f초의 %.0f%% 이상이 위험이면 경보 (%s)"
      % (DWELL_SEC, DWELL_RATIO * 100, DWELL_MODE))


def letterbox(img, size):
    """비율을 지키며 정사각형으로 맞추고 남는 곳은 회색으로 채운다."""
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh))
    px, py = (size - nw) // 2, (size - nh) // 2
    padded = cv2.copyMakeBorder(resized, py, size - nh - py, px, size - nw - px,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, px, py


def motion_pct(gray, prev):
    """직전 프레임과 비교해 '변한 픽셀' 이 몇 % 인지 돌려준다.

    작게 줄인 흑백 이미지로 계산하므로 1ms 남짓이면 끝난다.
    조명이 살짝 흔들리는 것까지 움직임으로 세지 않도록 25 이상 차이만 센다.
    """
    d = cv2.absdiff(gray, prev)
    n = int(np.count_nonzero(d > 25))
    return 100.0 * n / d.size


def to_gate_gray(frame):
    """게이팅용으로 작게 줄인 흑백 이미지를 만든다."""
    h, w = frame.shape[:2]
    g = cv2.resize(frame, (GATE_SIZE, max(1, int(GATE_SIZE * h / w))))
    return cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)


def detect(frame):
    """한 프레임에서 물체를 찾아 [(x1,y1,x2,y2,score,cls), ...] 로 돌려준다."""
    with span("preproc"):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb, r, px, py = letterbox(rgb, IMG_SIZE)

    with span("normalize"):
        if IN_INT:
            s, z = in_q[0], in_q[1]
            x = (lb.astype(np.float32) / 255.0 / (s if s else 1.0) + z)
            x = np.clip(x, -128, 127).astype(inp["dtype"])
        else:
            x = lb.astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        if NCHW:
            x = np.transpose(x, (0, 3, 1, 2))

    with span("inference"):
        interpreter.set_tensor(in_idx, x)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_idx)[0]

    if OUT_INT:
        s, z = out_q[0], out_q[1]
        raw = (raw.astype(np.float32) - z) * s

    tr.enter("postproc") if tr else None
    raw = raw.transpose()                      # (4+nc, N) -> (N, 4+nc)

    scores_all = raw[:, 4:]
    conf = scores_all.max(axis=1)
    cls = scores_all.argmax(axis=1)
    m = conf > CONF_TH
    if not m.any():
        tr.leave("postproc") if tr else None
        return []

    f = raw[m]
    sc, cl = conf[m], cls[m]
    cx, cy, w, h = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=-1)

    keep = cv2.dnn.NMSBoxesBatched(boxes, sc, cl,
                                   score_threshold=CONF_TH, nms_threshold=IOU_TH)
    res = []
    H, W = frame.shape[:2]
    for i in np.array(keep).flatten():
        bx, by, bw, bh = boxes[i] * IMG_SIZE
        x1 = int(np.clip((bx - px) / r, 0, W))
        y1 = int(np.clip((by - py) / r, 0, H))
        x2 = int(np.clip((bx + bw - px) / r, 0, W))
        y2 = int(np.clip((by + bh - py) / r, 0, H))
        res.append((x1, y1, x2, y2, float(sc[i]), int(cl[i])))
    tr.leave("postproc") if tr else None
    return res


def box_gap(a, b):
    """두 박스 사이의 최단 거리(픽셀). 겹치면 0."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx * dx + dy * dy) ** 0.5


def judge(dets):
    """위험 판정. (위험한가, 사유, 그릴 위험원 목록)"""
    babies = [d for d in dets if d[5] == SUBJECT]
    dangers = [d for d in dets if d[5] in DANGER]
    circles = []
    hit = None
    for dg in dangers:
        x1, y1, x2, y2, _, c = dg
        radius = (x2 - x1) * DANGER_SCALE
        circles.append((x1, y1, x2, y2, radius))
        for bb in babies:
            gap = box_gap(bb[:4], dg[:4])
            if gap <= radius:
                hit = (NAMES[c], gap, radius)
    return hit is not None, hit, circles


# ─────────────────────────────────────────────────────────────
#  카메라
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, CAM_BUFFER)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit(1)

if SHOW_GUI:
    cv2.namedWindow("baby-danger", cv2.WINDOW_NORMAL)

if RUN_SEC:
    print("\n%.0f초 동안 돌린 뒤 자동으로 끝냅니다." % RUN_SEC)
else:
    print("\n시작합니다. 화면 창을 누른 상태에서 q 를 누르면 종료.")
print("(--no-gui 일 때는 Ctrl+C)\n")

t_run0 = time.time()
n_frame = 0
# 움직임 게이팅에서 쓰는 값들
gate_prev = None        # 직전 프레임(작게 줄인 흑백)
prev_dets = []          # 건너뛸 때 재사용할 직전 검출 결과
prev_danger = False     # 직전 프레임이 위험이었는가
n_skip = 0              # 추론을 건너뛴 프레임 수
n_skip_run = 0          # 연속으로 건너뛴 수
mv_log = []             # 프레임마다의 '변한 픽셀 %'. 문턱값을 데이터로 정하려고 모은다
cpu_log = []

# ── 전력 측정 ──────────────────────────────────────────────
#   라즈베리파이 5 의 전원관리칩(PMIC)이 각 전원 레일의 전압과 전류를 알려준다.
#       vcgencmd pmic_read_adc
#   전압 x 전류 를 전부 더하면 보드가 쓰는 와트가 나온다. 추가 장비가 필요 없다.
#
#   한계 : 메인 5V 레일의 '전류' 는 PMIC 가 못 잰다. 그래서 절대값은 실제보다 낮다.
#          다만 빠지는 몫은 조건이 바뀌어도 거의 일정하므로
#          "FP32 대 INT8", "게이팅 켬 대 끔" 같은 **차이** 비교에는 유효하다.
#
#   측정이 루프를 느리게 만들면 재려던 값 자체가 망가지므로 별도 스레드에서 읽는다.
pw_log = []


def read_power():
    """PMIC 에서 지금 소비 전력(W)을 읽는다. 못 읽으면 None."""
    import subprocess, re
    try:
        out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None
    amps, volts = {}, {}
    for m in re.finditer(r"(\S+?)_([AV])\s+\w+\(\d+\)=([\d.]+)[AV]", out):
        name, kind, val = m.group(1), m.group(2), float(m.group(3))
        (amps if kind == "A" else volts)[name] = val
    if not amps:
        return None
    # 전류를 잰 레일마다 같은 이름의 전압을 곱해서 더한다
    w = sum(a * volts[n] for n, a in amps.items() if n in volts)
    return w if w > 0 else None


def power_worker(stop_flag):
    """1초마다 전력을 읽어 pw_log 에 쌓는다. 루프 타이밍을 건드리지 않는다."""
    while not stop_flag[0]:
        w = read_power()
        if w:
            pw_log.append(w)
        time.sleep(1.0)


_pw_stop = [False]
if read_power() is not None:
    import threading
    threading.Thread(target=power_worker, args=(_pw_stop,), daemon=True).start()
    print("[전력] PMIC 에서 소비 전력을 1초마다 기록합니다")
else:
    print("[전력] vcgencmd 로 전력을 읽지 못했습니다 (라즈베리파이 5 에서만 됩니다)")
try:
    import psutil
    psutil.cpu_percent(interval=None)
    psutil_ok = True
except ImportError:
    psutil_ok = False


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except OSError:
        return None


t_prev = time.time()
fps = 0.0
danger_since = None
last_alarm = 0.0
n_alarm = 0

# 최근 DWELL_SEC 구간의 (시각, 위험여부). 비율 판정에 쓴다.
from collections import deque
dwell_hist = deque()

# 두 방식을 한 번에 세어 Before/After 를 뽑는다.
#   실제 동작은 DWELL_MODE 쪽을 따르고, 다른 쪽은 "울렸을 것"만 센다.
n_alarm_cont = 0        # 연속 방식이었다면 울렸을 횟수
n_alarm_ratio = 0       # 비율 방식이었다면 울렸을 횟수
last_cont = 0.0
last_ratio = 0.0
n_danger_frame = 0      # 위험으로 판정된 프레임 수
# 비율 판정은 '최근 0.8초' 를 보고 울리므로, 경보가 나는 그 프레임에는
# 정작 아무것도 안 잡혀 있을 수 있다(info 가 None). 그때 쓸 마지막 정보를 남긴다.
last_info = None
n_reset = 0             # 연속 방식에서 타이머가 0 으로 돌아간 횟수

try:
    while True:
        with span("frame"):
            t_frame0 = time.time()      # 쉬는 시간을 계산할 때 쓴다

            with span("camera"):
                ok, frame = cap.read()
                if ok and MIRROR:
                    frame = cv2.flip(frame, 1)      # 좌우 반전 (거울 보기)
            if not ok:
                break

            n_frame += 1
            # 클립을 저장할 때 쓸 재생 속도를 실제 측정값으로 맞춘다.
            #   고정 20 FPS 로 저장하면 실제가 7 FPS 일 때 2.8배 빨리 감긴다.
            if clipper and n_frame % 30 == 0 and fps > 1.0:
                clipper.fps = fps
            if RUN_SEC and time.time() - t_run0 >= RUN_SEC:
                break
            if psutil_ok and n_frame % 20 == 0:
                cpu_log.append(psutil.cpu_percent(interval=None))

            # ── 움직임 게이팅 ──────────────────────────────
            #   화면이 안 변했고, 지금 위험 상태가 아니면 추론을 건너뛴다.
            #   초음파 센서로 하려던 "뭔가 있을 때만 추론" 을 소프트웨어로 한 것.
            skip = False
            if GATE:
                with span("gate"):
                    g = to_gate_gray(frame)
                    if gate_prev is not None:
                        mv = motion_pct(g, gate_prev)
                        mv_log.append(mv)
                        # 정지 + 안전 + 너무 오래 안 봤으면 아님 -> 건너뛴다
                        if (mv < GATE_TH and not prev_danger
                                and n_skip_run < GATE_MAX_SKIP):
                            skip = True
                    gate_prev = g

            if skip:
                n_skip += 1
                n_skip_run += 1
                dets = prev_dets          # 직전 결과를 그대로 쓴다. 비우지 않는다
            else:
                n_skip_run = 0
                # detect() 안에서 preproc / normalize / inference / postproc 구간이 기록된다
                dets = detect(frame)
                prev_dets = dets

            with span("judge"):
                is_danger, info, circles = judge(dets)
            prev_danger = is_danger     # 위험 중이면 다음 프레임은 절대 안 건너뛴다
            now = time.time()

            if tr:
                tr.counter("state", {"danger": 1 if is_danger else 0,
                                     "baby": sum(1 for d in dets if d[5] == SUBJECT)})

            # ── 체류 판정 ── 잠깐 스친 것으로 울리지 않게 한다.
            #   두 방식을 **둘 다** 계산한다. 실제 동작은 DWELL_MODE 쪽을 따르고,
            #   나머지는 "그 방식이었다면 울렸을 횟수"만 세어 비교에 쓴다.
            if is_danger:
                n_danger_frame += 1
                last_info = info        # 가장 최근에 실제로 본 위험

            # (가) 연속 방식 — 한 프레임만 놓쳐도 처음부터 다시
            if is_danger:
                if danger_since is None:
                    danger_since = now
            else:
                if danger_since is not None:
                    n_reset += 1        # 위험이 이어지다 끊긴 횟수
                danger_since = None
            alarm_cont = (danger_since is not None and now - danger_since >= DWELL_SEC)

            # (나) 비율 방식 — 최근 창에서 몇 %가 위험이었나
            dwell_hist.append((now, is_danger))
            while dwell_hist and now - dwell_hist[0][0] > DWELL_SEC:
                dwell_hist.popleft()
            n_win = len(dwell_hist)
            n_hit = sum(1 for _, d in dwell_hist if d)
            # 창이 충분히 차야 판정한다. 시작 직후 한두 장으로 울리면 안 된다.
            #   예전에는 "창 안의 가장 오래된 프레임이 0.4초 이상 전" 을 조건으로 했는데,
            #   FPS 가 낮으면(7.6 FPS = 132ms 간격) 창에 4장뿐이라 그 조건이
            #   0.396초에서 아슬아슬하게 실패해 **판정 자체가 열리지 않았다.**
            #   그래서 연속 방식보다 오히려 덜 울리는 역전이 생겼다.
            #   프로그램이 DWELL_SEC 이상 돌았는지로 바꾸면 FPS 와 무관해진다.
            win_full = (n_win >= 3 and now - t_run0 >= DWELL_SEC)
            alarm_ratio = win_full and (n_hit / n_win) >= DWELL_RATIO

            # 비교용 집계 (쿨다운은 각자 따로 센다)
            if alarm_cont and now - last_cont > COOLDOWN_SEC:
                last_cont = now
                n_alarm_cont += 1
            if alarm_ratio and now - last_ratio > COOLDOWN_SEC:
                last_ratio = now
                n_alarm_ratio += 1

            alarming = alarm_cont if DWELL_MODE == "continuous" else alarm_ratio

            # 클립에 담기 — 검출 결과가 나온 뒤라야 박스를 그릴 수 있다.
            # 화면에 그리기 전에 담으므로 원본은 그대로다.
            if clipper:
                clipper.push(frame, dets, alarming)

            if alarming and now - last_alarm > COOLDOWN_SEC:
                last_alarm = now
                n_alarm += 1
                if tr:
                    tr.instant("ALARM")          # 타임라인에 세로 표시가 찍힌다
                play_alarm()
                _ii = info or last_info
                if _ii:
                    print("[경보 %d] %s 에 접근  (거리 %.0f px / 반경 %.0f px)"
                          % (n_alarm, _ii[0], _ii[1], _ii[2]))
                else:
                    print("[경보 %d] 위험 상태" % n_alarm)

                # 보호자 휴대폰으로. 사진은 박스를 그린 뒤라야 뜻이 있으므로
                # 여기서 사본에 그려서 보낸다 (원본 frame 은 건드리지 않는다).
                #
                # ** 알림·클립에서 무슨 일이 나든 감시는 계속되어야 한다. **
                #   알림 하나 못 보낸 것보다 감시가 멈추는 쪽이 훨씬 큰 사고다.
                #   실제로 notify.py 가 옛 버전이던 보드에서 경보 직후 프로그램이
                #   통째로 죽어(웹 화면까지 같이 끊겼다) 이 감싸기를 넣었다.
                try:
                    if noti:
                        shot = frame.copy()
                        for (bx1, by1, bx2, by2, bsc, bc) in dets:
                            cv2.rectangle(shot, (bx1, by1), (bx2, by2),
                                          COLORS.get(bc, (255, 255, 255)), 2)
                            cv2.putText(shot, "%s %d%%" % (NAMES[bc], int(bsc * 100)),
                                        (bx1, max(12, by1 - 6)),
                                        cv2.FONT_HERSHEY_PLAIN, 1,
                                        COLORS.get(bc, (255, 255, 255)), 2)
                        cv2.rectangle(shot, (0, 0),
                                      (shot.shape[1] - 1, shot.shape[0] - 1),
                                      (0, 0, 255), 6)
                        _t = ("아기가 %s 에 접근했습니다. (거리 %.0f px / 위험반경 %.0f px)"
                              % (_ii[0], _ii[1], _ii[2])) if _ii else "아기가 위험물에 접근했습니다."
                        noti.send(_t, shot, click=WEB_URL)

                    # 영상은 뒤 구간을 더 모아야 하므로 여기서는 시작만 시킨다.
                    # 다 모이면 clip.py 가 알아서 묶어 보낸다.
                    if clipper:
                        clipper.trigger("아기가 %s 에 접근했습니다. (경보 전 3초 + 후 2초)"
                                        % (_ii[0] if _ii else "위험물"))
                except Exception as e:
                    print("[경보] 알림/영상 전송에서 문제가 있었지만 감시는 계속합니다 :", e)

            # LED
            with span("led"):
                if alarming:
                    set_led("red")
                elif any(d[5] == SUBJECT for d in dets):
                    set_led("mid")          # 아기는 보이지만 아직 위험하지 않다
                else:
                    set_led("green")

            # 웹으로 내보내기 — 박스를 그린 사본을 올린다.
            # 보는 사람이 없으면 web.py 가 JPEG 인코딩조차 건너뛴다(저전력).
            if web:
                state = "DANGER" if alarming else (
                    "DETECTED" if any(d[5] == SUBJECT for d in dets) else "SAFE")
                if web.n_view > 0:
                    shot = frame.copy()
                    for (bx1, by1, bx2, by2, brad) in circles:
                        cv2.circle(shot, ((bx1 + bx2) // 2, (by1 + by2) // 2),
                                   int(brad), (0, 140, 255), 1, cv2.LINE_AA)
                    for (bx1, by1, bx2, by2, bsc, bc) in dets:
                        bcol = COLORS.get(bc, (255, 255, 255))
                        cv2.rectangle(shot, (bx1, by1), (bx2, by2), bcol, 2)
                        cv2.putText(shot, "%s %d%%" % (NAMES[bc], int(bsc * 100)),
                                    (bx1, max(12, by1 - 6)),
                                    cv2.FONT_HERSHEY_PLAIN, 1, bcol, 2)
                    if alarming:
                        cv2.rectangle(shot, (0, 0),
                                      (shot.shape[1] - 1, shot.shape[0] - 1),
                                      (0, 0, 255), 6)
                        cv2.putText(shot, "DANGER", (10, 56),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    shot = frame
                web.publish(shot, state, n_alarm, fps,
                            [{"name": NAMES.get(d[5], d[5]), "conf": d[4]} for d in dets],
                            cpu_temp(),
                            ("%s / %.0fpx" % (info[0], info[1])) if info else None)

            # 화면
            if SHOW_GUI:
                with span("draw"):
                    for (x1, y1, x2, y2, rad) in circles:
                        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), int(rad),
                                   (0, 140, 255), 1, cv2.LINE_AA)
                    for (x1, y1, x2, y2, sc, c) in dets:
                        col = COLORS.get(c, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(frame, "%s %d%%" % (NAMES[c], int(sc * 100)),
                                    (x1, max(12, y1 - 6)),
                                    cv2.FONT_HERSHEY_PLAIN, 1, col, 2)

                    now2 = time.time()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now2 - t_prev))
                    t_prev = now2
                    cv2.putText(frame, "FPS %.1f   alarm %d" % (fps, n_alarm),
                                (10, 24), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 255), 2)
                    if alarming:
                        cv2.rectangle(frame, (0, 0),
                                      (frame.shape[1] - 1, frame.shape[0] - 1),
                                      (0, 0, 255), 6)
                        cv2.putText(frame, "DANGER", (10, 56),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                with span("imshow"):
                    cv2.imshow("baby-danger", frame)

                # 화면에 실제로 그려지는 일은 waitKey 안에서 일어난다.
                # 그래서 이 구간이 크면 '추론이 느린' 게 아니라 '표시가 느린' 것이다.
                with span("waitKey"):
                    quit_now = (cv2.waitKey(WAIT_MS) & 0xFF == ord("q"))
                if quit_now:
                    break
            else:
                now2 = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now2 - t_prev))
                t_prev = now2

            # ── 조용할 때는 쉰다 (게이팅 2단계) ────────────
            #   직전 프레임을 건너뛰었다면 = 아무 일도 안 일어나고 있다는 뜻.
            #   다음 프레임을 서둘러 읽을 이유가 없다.
            if GATE and GATE_IDLE_FPS > 0 and n_skip_run > 0:
                with span("idle"):
                    _target = 1.0 / GATE_IDLE_FPS
                    _used = time.time() - t_frame0
                    if _used < _target:
                        time.sleep(_target - _used)
                    # 자는 동안 카메라 버퍼에 쌓인 낡은 프레임을 버린다.
                    #   버리지 않으면 몇 백 ms 전 장면을 보고 판단하게 된다.
                    #   grab() 은 영상을 풀지 않아서 read() 보다 훨씬 싸다.
                    for _ in range(max(0, CAM_BUFFER - 1)):
                        cap.grab()
except KeyboardInterrupt:
    pass
finally:
    _pw_stop[0] = True
    cap.release()
    if SHOW_GUI:
        cv2.destroyAllWindows()
    for d in leds.values():
        d.off()
        d.close()

    # ── 종료 요약 ──────────────────────────────────────────
    total = time.time() - t_run0
    print()
    print("=" * 52)
    print(" 측정 요약   모델 %s" % os.path.basename(MODEL))
    print("=" * 52)
    print("  돌린 시간      : %.1f 초" % total)
    print("  처리 프레임    : %d" % n_frame)
    if total > 0:
        print("  평균 FPS       : %.1f" % (n_frame / total))
    if n_frame:
        print("  한 프레임      : %.1f ms" % (total * 1000.0 / n_frame))
    print("  경보 횟수      : %d   (판정 방식 %s)" % (n_alarm, DWELL_MODE))

    # ── 움직임 게이팅 결과 ──────────────────────────────
    #   FPS 만 보면 착각한다. 건너뛴 프레임은 1ms 만에 끝나므로 FPS 가 올라간다.
    #   실제로 아낀 것은 '추론 횟수' 다. 그래서 둘을 나눠 찍는다.
    if GATE:
        n_infer = n_frame - n_skip
        print()
        print("  ── 움직임 게이팅 (초음파 대체) ──")
        print("    문턱값        : 변한 픽셀 %.2f%% 미만이면 정지로 봄" % GATE_TH)
        print("    처리 프레임   : %d" % n_frame)
        print("    실제 추론     : %d 회" % n_infer)
        print("    건너뛴 프레임 : %d 회  (%.1f%%)"
              % (n_skip, 100.0 * n_skip / max(1, n_frame)))
        if n_frame:
            print("    -> 추론을 %.1f%% 줄였다" % (100.0 * n_skip / n_frame))
        if GATE_IDLE_FPS > 0:
            print("    조용할 때 목표 : %.0f FPS 로 쉬면서 돈다" % GATE_IDLE_FPS)
        else:
            print("    조용할 때 쉬지 않음 (--gate-idle-fps 8 로 켠다)")
        print("    ※ 게이팅을 켜면 FPS 가 올라간다(건너뛴 프레임이 빨라서).")
        print("       아낀 것은 속도가 아니라 '추론 횟수' 다. 전력은 이쪽을 따라간다.")

        # ── 변한 픽셀 % 의 분포 ────────────────────────
        #   문턱값(0.6%)을 추측으로 정하면 안 된다. 실제로 어떤 값이 나오는지 보고 정한다.
        #   같은 장면인데 실행마다 건너뛴 비율이 크게 달라지면 여기에 원인이 보인다.
        if mv_log:
            q = sorted(mv_log)
            def pct(p):
                return q[min(len(q) - 1, int(len(q) * p / 100.0))]
            print()
            print("    ── 변한 픽셀 %% 의 분포 (표본 %d) ──" % len(q))
            print("      최소 %.3f   25%% %.3f   중앙 %.3f   75%% %.3f   95%% %.3f   최대 %.3f"
                  % (q[0], pct(25), pct(50), pct(75), pct(95), q[-1]))
            for th in (0.2, 0.4, 0.6, 1.0, 2.0, 5.0):
                n = sum(1 for v in q if v < th)
                print("      문턱값 %.1f%% 이면 -> %5.1f%% 건너뜀" % (th, 100.0 * n / len(q)))
            print("      ※ 정지 상태인데 중앙값이 문턱값 근처면 판정이 불안정하다.")
            print("         자동노출·창밖 움직임·형광등 깜빡임이 원인일 수 있다.")
    else:
        print("  게이팅         : 꺼짐 (--gate 로 켠다)")

    # ── 전력 (라즈베리파이 5 의 PMIC 에서 직접 읽는다) ──
    if pw_log:
        pw_log.sort()
        print()
        print("  ── 전력 (PMIC 실측) ──")
        print("    평균          : %.2f W" % (sum(pw_log) / len(pw_log)))
        print("    최소 / 최대   : %.2f W / %.2f W" % (pw_log[0], pw_log[-1]))
        print("    표본          : %d 회" % len(pw_log))
        print("    ※ 메인 5V 레일 전류는 PMIC 가 못 잰다. 절대값은 실제보다 낮다.")
        print("       조건끼리 '차이' 를 비교하는 용도로 쓸 것.")
    print()
    print("  ── 체류 판정 방식 비교 (같은 장면, 같은 프레임) ──")
    print("    연속(예전)  : %d 회   <- 0.5초 동안 한 프레임도 안 놓쳐야 경보"
          % n_alarm_cont)
    print("    비율(지금)  : %d 회   <- 최근 0.5초의 %.0f%% 이상이면 경보"
          % (n_alarm_ratio, DWELL_RATIO * 100))
    print("    위험 프레임 : %d / %d" % (n_danger_frame, n_frame))
    print("    연속이 끊긴 횟수 : %d" % n_reset)
    if n_frame and total > 0:
        _fps = n_frame / total
        _win = _fps * DWELL_SEC
        print("    판정 창 : %.1f초 x %.1f FPS = 약 %.1f장" % (DWELL_SEC, _fps, _win))
        if _win < 6:
            print("      창이 %d장뿐이라 판정이 거칠다. 한 장 차이로 결과가 바뀐다." % round(_win))
            print("      --dwell %.1f 로 늘리거나 --dwell-ratio 0.4 로 낮춰 보라."
                  % max(0.8, 6.0 / max(_fps, 1.0)))
        if n_reset and n_danger_frame:
            print("    위험이 한 번에 이어진 평균 길이 : %.1f장"
                  % (n_danger_frame / max(n_reset, 1)))
            if n_danger_frame / max(n_reset, 1) < _win * 0.6:
                print("      검출이 띄엄띄엄 되고 있다. 문턱값(CONF_TH)을 낮추거나")
                print("      거리를 줄이면 이어지는 길이가 늘어난다.")
    if n_alarm_ratio > n_alarm_cont:
        print("    -> 연속 방식은 프레임을 놓칠 때마다 타이머가 0 으로 돌아가")
        print("       경보를 %d 회 놓쳤다." % (n_alarm_ratio - n_alarm_cont))
    if cpu_log:
        print("  CPU 평균       : %.1f %%" % (sum(cpu_log) / len(cpu_log)))
    t = cpu_temp()
    if t:
        print("  CPU 온도       : %.1f C" % t)
    print("=" * 52)

    if web:
        print(web.summary())

    # 정리하는 동안 Ctrl+C 를 한 번 더 누르면 빨간 오류가 뜬다.
    # 동작에는 문제가 없지만 시연 중에 보기 안 좋으므로 조용히 넘긴다.
    if clipper:
        print(clipper.summary())
        try:
            t_wait = time.time()
            while clipper._busy and time.time() - t_wait < 15:   # 만들던 클립 마무리
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("  (클립 마무리를 기다리지 않고 끝냅니다)")

    if noti:
        print(noti.summary())
        try:
            noti.close()   # 큐에 남은 것을 마저 보내고 정리
        except KeyboardInterrupt:
            print("  (보내던 알림을 기다리지 않고 끝냅니다)")
        except Exception:
            pass

    if tr:
        tr.save()          # 단계별 평균 표 출력 + trace_baby.json 저장
