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

세 가지 상태 (저전력) — 2026-09-26 3상태 판 (팀원A 두 모드 판을 고친 것)
    대기(standby) : 화면에 아기가 없다. STANDBY_INTERVAL(5초)마다 한 장만 추론.
                    모델 = SSDLite320-MobileNetV3 (FP32)
    감시(active)  : 아기는 있지만 위험 아님. YOLO11n 을 쓰되
                    - 움직임 게이팅: 화면이 안 변하면 추론을 건너뛴다 (기본 켬, --no-gate 로 끔)
                    - 속도 상한: 루프를 WATCH_FPS(10) 이상으로 돌리지 않는다
    경보(alert)   : 아기가 위험 반경 안. 매 프레임 전속력, 게이팅·상한 없음.
    전이
        대기 -> 감시 : 아기가 보이면 즉시
        감시 -> 경보 : 위험 판정이 나면 즉시
        경보 -> 감시 : 위험이 HOLD_SEC(3초) 동안 없으면
        감시 -> 대기 : 아기가 HOLD_SEC(3초) 동안 안 보이면 (위험물이 보이든 말든)
    왜 고쳤나
        두 모드 판은 "아기+위험물이 함께 안 보이면 3초 뒤 대기" 라서, 아기만 있을 때
        대기(5초) <-> 고성능(3초) 를 무한 왕복했다 (9/26 실측: 45초에 6번 왕복, 4.21 W).
        그 사이 5초짜리 감시 공백도 생겼다. 감시 상태를 따로 두면 아기가 가만히 있을 때는
        게이팅으로 쉬고(1.8 W), 움직이면 상한 FPS 로 보고, 위험해지면 즉시 전속력이 된다.

추론 엔진
    LiteRT (pip install ai-edge-litert) 를 **반드시** 쓴다. XNNPACK 가속이 자동으로 붙는다.
    예전 tflite-runtime 2.15 는 같은 모델도 느리게 돌려서(특히 INT8) 쓰지 않는다.

전력 기록 (라즈베리파이 5)
    1초마다 PMIC 전력을 읽어 CSV 로 저장한다. 파일은 실행할 때마다 새로 만든다.
        power_log_<날짜_시각>.csv      1초마다 한 줄 (시각, 경과, W, 모드, 누적 Wh, 온도)
        power_summary_<날짜_시각>.csv  끝날 때 모드별 요약 (시간, 평균 W, Wh, 시간당 Wh)

실행
    source ~/work/env/bin/activate
    cd ~/work/baby
    python3 detect_baby.py                      # 기본 모델 두 개 (아래 이름)
    python3 detect_baby.py --standby-model ssdlite320_mbv3_fp32.tflite --active-model yolo11n_fp32.tflite
    python3 detect_baby.py --standby-interval 5 # 대기 모드 추론 간격(초)
    python3 detect_baby.py --hold 3             # 고성능 모드 유지 시간(초)
    python3 detect_baby.py --power-csv run1     # CSV 이름 앞부분 지정 (run1_log.csv / run1_summary.csv)
    python3 detect_baby.py --no-gui             # 화면 없이 (측정용)
    python3 detect_baby.py --trace              # 단계별 시간 측정 + 타임라인 기록
    python3 detect_baby.py --notify             # 경보 시 보호자 휴대폰으로 사진 전송
    python3 detect_baby.py --notify --clip      # 사진 + 경보 앞뒤 5초 영상까지
    python3 detect_baby.py --clip-raw           # 영상에 박스를 안 그림 (날것)
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

# ── 추론 엔진 : LiteRT 필수 ─────────────────────────────────
#   같은 .tflite 파일을 PC 에서 5라운드 비교한 결과 (README 트러블 슈팅)
#       FP32 : tflite-runtime 2.15 -> LiteRT 2.2 에서 1.05~1.45배 빨라짐
#       INT8 : 2.3~2.8배 빨라짐 (2.15 는 INT8 을 제대로 가속하지 못한다)
#   예전 엔진으로 조용히 넘어가면 같은 모델인데 느려지고 전력 측정값도 달라진다.
#   그래서 대체하지 않고, 없으면 설치 안내를 띄우고 끝낸다.
try:
    import ai_edge_litert
    import ai_edge_litert.interpreter as tflite
    ENGINE = "LiteRT %s" % getattr(ai_edge_litert, "__version__", "")
except ImportError:
    print("추론 엔진 LiteRT 가 없습니다. 설치 후 다시 실행하세요 :")
    print("    pip install ai-edge-litert")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────────────────────
def _argval(flag, default=""):
    """--flag 값  형태의 인자를 읽는다."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# 모델 두 개 — README.md 의 모델 선정 결과 (5개 모델 비교, epoch 120 / 회전 ±10°)
#
#   대기 모드   SSDLite320-MobileNetV3 (FP32)
#       찾기(AP50) 0.984 로 1위, 연산량 0.82 GFLOPs 로 최소, 속도는 YOLO11n 의 0.55배.
#       대기 모드는 '아기가 있는가' 만 보면 되므로 찾기 성능이 기준이다.
#   고성능 모드 YOLO11n (FP32)
#       knife 위치 정확도(AP50-95) 0.748 로 1위. 아기-위험물 거리를 재야 하므로
#       박스 위치가 정확한 모델을 쓴다.
#
#   파일 이름이 다르면 --standby-model / --active-model 로 지정한다.
STANDBY_MODEL = "ssdlite320_mbv3_fp32.tflite"
ACTIVE_MODEL = "yolo11n_fp32.tflite"
CONF_TH = 0.40
IOU_TH = 0.45
THREADS = int(_argval("--threads", "4"))   # 추론 스레드 수 (저전력 실험: 4 vs 2)
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


# ── 검출 문턱값을 명령줄에서 바꾼다 ────────────────────────
#   이 값이 '정확도' 를 좌우하는 거의 유일한 설정이다.
#   몇 % 이상을 "찾았다" 로 칠 것인가.
#
#     낮추면 (0.25)  멀리 있는 칼도 잡는다.  대신 책·옷도 칼로 잡는다(오검출)
#     올리면 (0.50)  헛것을 덜 잡는다.      대신 먼 칼을 놓친다
#
#   전처리(letterbox + /255)와 NMS 는 학습 때와 맞춰져 있어 건드릴 것이 없다.
#   즉 같은 모델이라면 코드에서 결과를 바꿀 수 있는 조절 항목는 여기뿐이다.
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
GATE = "--no-gate" not in sys.argv    # 3상태 판: 감시 상태 게이팅 기본 켬. --no-gate 로 끔 (--gate 는 호환용)
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

# ── 두 모드 설정 ─────────────────────────────────────────────
STANDBY_MODEL = _argval("--standby-model", STANDBY_MODEL)
ACTIVE_MODEL = _argval("--active-model", ACTIVE_MODEL)
# 대기 모드에서 몇 초마다 한 장을 추론할지. README 설계값 5초.
#   길수록 전력은 줄지만, 아기가 화면에 들어온 뒤 알아채기까지 최대 이만큼 늦어진다.
STANDBY_INTERVAL = float(_argval("--standby-interval", "5"))
# 고성능 모드 유지 시간. 아기와 위험물이 '함께' 보이지 않는 상태가
#   이 시간 이어지면 대기 모드로 돌아간다. 한두 프레임 놓쳤다고 바로 빠지지 않게 한다.
HOLD_SEC = float(_argval("--hold", "3"))
# 전력 CSV 파일 이름 앞부분. 비우면 power_log_<시각>.csv / power_summary_<시각>.csv
POWER_CSV = _argval("--power-csv", "")
# 감시 상태의 루프 상한 FPS. 아기가 있어도 위험이 아니면 이 속도 이상으로는 돌지 않는다.
#   체류 판정이 0.5초 창에 3장 이상을 원하므로 6 이상이어야 한다. 0 이면 상한 없음.
WATCH_FPS = float(_argval("--watch-fps", "5"))
# ── 4상태 판 추가 옵션 ──────────────────────────────────────
#   경계(guard) : 아기와 위험물이 같이 보이고, 거리가 위험 반경의 GUARD_SCALE 배 안일 때.
#                 감시보다 빠르게(GUARD_FPS) 보고, 고해상도 모델(--hi-model)이 있으면 그걸 쓴다.
#   경보(alert) : 위험 반경 안. 전속력 + 고해상도 모델.
#   클럭        : --clock-low/--clock-high 를 주면 대기·감시에서는 low, 경계·경보에서는 high 로
#                 CPU 최대 클럭을 바꾼다 (kHz, 예: 1500000 / 2400000). sudo 가 비밀번호 없이 되어야 한다.
GUARD_FPS = float(_argval("--guard-fps", "15"))
# 경계 상태에서도 움직임 게이팅을 쓸지 (--guard-gate). 아기가 위험물 근처에 있지만 가만히 있으면
#   추론을 건너뛴다. 안전 장치(위험 중 건너뛰기 금지, 연속 GATE_MAX_SKIP 장이면 강제 추론)는 같다.
GUARD_GATE = "--guard-gate" in sys.argv
# 2D 거리 문제 완화 (선택): 위험물 박스 폭 / 아기 박스 높이 가 R 미만이면 위험물이 훨씬 멀리(뒤쪽) 있다고
#   보고 그 쌍은 판정에서 뺀다. 칼 폭 15 cm / 아기 키 60~70 cm ≈ 0.2 가 같은 거리일 때의 비율.
#   칼이 세로로 놓이면 폭이 작아져 오판할 수 있으므로 보수적으로(예: 0.1) 쓰거나 끈다. 0 = 끔.
DEPTH_CHECK = float(_argval("--depth-check", "0"))
n_depth_skip = [0]
# 알림 피로 완화: 같은 경보가 이어지면 휴대폰 알림을 첫 번째는 바로, 두 번째는 30초 뒤(진짜 계속되는지 확인),
#   그 뒤로는 5분(300초)에 한 번만 보낸다. 경보 상태를 벗어나면 다시 처음부터.
#   보드의 LED·소리·웹 화면은 그대로 매 경보마다 동작한다. --notify-gaps 30,300 처럼 바꿀 수 있다.
ALARM_BACKOFF = "--no-alarm-backoff" not in sys.argv
NOTIFY_GAPS = [float(x) for x in _argval("--notify-gaps", "30,300").split(",")]
notify_streak = [0]        # 이번 경보 구간에서 실제로 보낸 알림 수
last_notify_t = [0.0]
n_notify_skip = [0]
GUARD_SCALE = float(_argval("--guard-scale", "2.0"))
HI_MODEL = _argval("--hi-model", "")
CLOCK_LOW = _argval("--clock-low", "")
CLOCK_HIGH = _argval("--clock-high", "")
_CPU_POL = "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"
_clock_now = [None]
n_clock_sw = [0]


def set_clock(khz):
    """CPU 최대 클럭을 바꾼다 (kHz 문자열). 실패해도 감시는 계속한다."""
    if not khz or _clock_now[0] == khz:
        return
    import subprocess
    try:
        subprocess.run(["sudo", "-n", "tee", _CPU_POL], input=(khz + "\n").encode(),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=True)
        _clock_now[0] = khz
        n_clock_sw[0] += 1
    except Exception as e:
        print("[클럭] 바꾸지 못했습니다 (%s) : %s" % (khz, e))


_clock_default = ""
if CLOCK_LOW or CLOCK_HIGH:
    try:
        _clock_default = open(_CPU_POL.replace("scaling_max_freq", "cpuinfo_max_freq")).read().strip()
    except OSError:
        pass
    set_clock(CLOCK_LOW or _clock_default)
# 경보 상태 루프 상한 FPS (0 = 상한 없음 = 전속력). 저전력 실험용: INT8 처럼 추론이 빠르면
#   상한 없이 두면 카메라 30 FPS 까지 그냥 더 많이 돌아서 전력이 안 준다.
ALERT_FPS = float(_argval("--alert-fps", "0"))
# 대기 상태에서 카메라를 놓았다가(release) 5초마다 다시 연다. USB 웹캠 자체 전력을 줄이려는 시도.
#   PMIC 값에는 안 잡히므로 USB-C 전력계로 봐야 한다. 다시 여는 데 걸리는 시간을 요약에 찍는다.
CAM_OFF_STANDBY = "--cam-off-standby" in sys.argv
# 아무것도 안 돌릴 때의 전력(W). 단계별 에너지 추정의 기준선. 9/26 지인 보드 1.91 W.
IDLE_W = float(_argval("--idle-w", "1.91"))

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
#   (예전에는 --conf / --dwell 등이 빠져 있어서 --conf 0.25 의 0.25 가 모델 이름으로 잡혔다)
_VALUE_FLAGS = ("--web-port", "--web-token", "--seconds", "--web-url", "--conf", "--iou",
                "--gate-th", "--gate-max", "--gate-idle-fps", "--dwell", "--dwell-ratio",
                "--standby-model", "--active-model", "--standby-interval", "--hold", "--power-csv",
                "--watch-fps", "--threads", "--alert-fps", "--idle-w",
                "--guard-fps", "--guard-scale", "--hi-model", "--clock-low", "--clock-high",
                "--depth-check", "--notify-gaps")
_skip = set()
for _f in _VALUE_FLAGS:
    if _f in sys.argv:
        _i = sys.argv.index(_f)
        if _i + 1 < len(sys.argv):
            _skip.add(_i + 1)
args = [a for i, a in enumerate(sys.argv)
        if i >= 1 and not a.startswith("--") and i not in _skip]
if args:
    # 예전처럼 모델 하나만 적으면 고성능 모드 모델로 쓴다
    ACTIVE_MODEL = args[0]
MODEL = ACTIVE_MODEL      # 웹 화면 등 '대표 모델 이름' 이 필요한 곳에서 쓴다

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


STAGE_T = {}        # 단계 이름 -> 누적 초. 트레이스를 안 켜도 모은다 (perf_counter 두 번, ~1 us)
STAGE_N = {}


class _Acc:
    """단계 시간을 누적하는 껍데기. 트레이서가 있으면 그 span 도 같이 연다."""
    __slots__ = ("name", "inner", "t0")

    def __init__(self, name, inner):
        self.name, self.inner = name, inner

    def __enter__(self):
        self.t0 = time.perf_counter()
        if self.inner is not None:
            self.inner.__enter__()
        return self

    def __exit__(self, *a):
        if self.inner is not None:
            self.inner.__exit__(*a)
        STAGE_T[self.name] = STAGE_T.get(self.name, 0.0) + (time.perf_counter() - self.t0)
        STAGE_N[self.name] = STAGE_N.get(self.name, 0) + 1
        return False


def span(name, tid=1):
    return _Acc(name, tr.span(name, tid) if tr else None)


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
                               min_gap_sec=30.0, click=WEB_URL)
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
#  모델 두 개 (대기 모드 SSDLite / 고성능 모드 YOLO11n)
# ─────────────────────────────────────────────────────────────
for _m in (STANDBY_MODEL, ACTIVE_MODEL):
    if not os.path.exists(_m):
        print("모델 파일이 없습니다:", _m)
        print("  --standby-model / --active-model 로 경로를 지정하세요.")
        sys.exit(1)


class Detector:
    """tflite 모델 하나를 감싼다. YOLO 와 SSDLite 를 같은 방식으로 부를 수 있게 한다.

    모델 형식은 출력 텐서 모양을 보고 스스로 판별한다.
        yolo : 출력 1개 (1, 4+클래스, N)   cx cy w h + 클래스별 점수  -> NMS 필요
               입력은 letterbox(비율 유지 + 회색 여백). 좌표는 0~1 정규화.
        ssd  : 출력 2개 boxes (1, N, 4) + scores (1, N, 클래스)
               - boxes  : x1 y1 x2 y2, 0~1 정규화 (앵커 디코딩은 모델 안에서 끝남)
               - scores : 클래스별 확률 (배경은 이미 빠져 있음)
               입력은 **letterbox 없이** 320x320 으로 늘려서 넣는다(학습 때와 같게).
               정규화((x-0.5)/0.5)는 모델 안에 들어 있으므로 0~1 만 맞추면 된다.
               NMS 는 가변 길이라 모델 밖에서 한다(여기서).
    """

    def __init__(self, path, name):
        self.path, self.name = path, name
        try:
            self.itp = tflite.Interpreter(model_path=path, num_threads=THREADS)
        except TypeError:
            self.itp = tflite.Interpreter(model_path=path)
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()[0]
        self.outs = self.itp.get_output_details()
        shape = list(self.inp["shape"])
        # 입력이 NCHW 인지 NHWC 인지 자동 판별 (Ultralytics / torchvision 변환본은 NCHW)
        self.nchw = (shape[1] == 3)
        self.size = shape[2] if self.nchw else shape[1]
        self.kind = "ssd" if len(self.outs) >= 2 else "yolo"
        self.in_int = self.inp["dtype"] in (np.int8, np.uint8)
        self.in_q = self.inp.get("quantization", (0.0, 0))
        self.box_idx = None       # ssd : 출력 2개 중 어느 쪽이 boxes 인지 (첫 추론 때 판별)
        self.normalized = None    # yolo : 좌표가 0~1 인지 (확실한 박스가 처음 나올 때 판별)
        self.n_infer = 0
        self.t_infer = 0.0
        print("[%s] %s" % (name, path))
        print("      형식 %s / 입력 %s %s / dtype %s / 출력 %s" % (
            self.kind, shape, "NCHW" if self.nchw else "NHWC", np.dtype(self.inp["dtype"]).name,
            " + ".join(str(list(o["shape"])) for o in self.outs)))

    # ---------- 전처리 ----------
    def _prep(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        if self.kind == "ssd":
            img = cv2.resize(rgb, (self.size, self.size))
            meta = (self.size / W, self.size / H, 0, 0)          # (x배율, y배율, x여백, y여백)
        else:
            img, r, px, py = letterbox(rgb, self.size)
            meta = (r, r, px, py)
        x = img.astype(np.float32) / 255.0
        if self.in_int:                                            # 입력까지 정수인 모델 대비
            s, z = self.in_q
            info = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(x / (s if s else 1.0) + z), info.min, info.max).astype(self.inp["dtype"])
        x = np.expand_dims(x, 0)
        if self.nchw:
            x = np.transpose(x, (0, 3, 1, 2))
        return np.ascontiguousarray(x), meta

    def _outputs(self):
        res = []
        for d in self.outs:
            o = self.itp.get_tensor(d["index"])
            if d["dtype"] in (np.int8, np.uint8):                 # 출력이 정수면 실수로 환산
                s, z = d.get("quantization", (0.0, 0))
                o = (o.astype(np.float32) - z) * s
            res.append(o[0])
        return res

    # ---------- 한 프레임 검출 ----------
    def __call__(self, frame):
        """[(x1,y1,x2,y2,score,cls), ...] (원본 프레임 픽셀 좌표)"""
        with span("preproc"):
            x, meta = self._prep(frame)
        with span("inference"):
            t0 = time.time()
            self.itp.set_tensor(self.inp["index"], x)
            self.itp.invoke()
            outs = self._outputs()
            self.t_infer += time.time() - t0
            self.n_infer += 1
        with span("postproc"):
            return self._post(outs, meta, frame.shape[:2])

    def _post(self, outs, meta, hw):
        S = self.size
        if self.kind == "yolo":
            raw = outs[0]
            if raw.shape[0] < raw.shape[1]:          # (4+nc, N) -> (N, 4+nc)
                raw = raw.T
            sc_all = raw[:, 4:]
            conf, cls = sc_all.max(axis=1), sc_all.argmax(axis=1)
            b = raw[:, :4].copy()
            # 좌표 단위 판별 : Ultralytics 변환본은 0~1 정규화가 보통이지만 확실히 하기 위해
            #   점수가 높은(진짜) 박스만 보고 한 번 정한다. 점수≈0 인 행은 값이 튈 수 있어 보지 않는다.
            if self.normalized is None and (conf >= CONF_TH).any():
                self.normalized = bool(np.abs(b[conf >= CONF_TH]).max() <= 2.0)
            if self.normalized is not False:
                b *= S
            b = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                          b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
        else:
            if self.box_idx is None:                 # x2>=x1, y2>=y1 이 성립하는 쪽이 boxes
                ok = [((a[:, 2] >= a[:, 0]) & (a[:, 3] >= a[:, 1])).mean() for a in outs[:2]]
                self.box_idx = int(np.argmax(ok))
            b = outs[self.box_idx] * S
            sc_all = outs[1 - self.box_idx]
            conf, cls = sc_all.max(axis=1), sc_all.argmax(axis=1)

        m = conf > CONF_TH
        if not m.any():
            return []
        b, conf, cls = b[m], conf[m], cls[m]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
        keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(),
                                       score_threshold=CONF_TH, nms_threshold=IOU_TH)
        rx, ry, px, py = meta
        H, W = hw
        res = []
        for i in np.array(keep).flatten():
            x1, y1, x2, y2 = b[i]
            res.append((int(np.clip((x1 - px) / rx, 0, W)), int(np.clip((y1 - py) / ry, 0, H)),
                        int(np.clip((x2 - px) / rx, 0, W)), int(np.clip((y2 - py) / ry, 0, H)),
                        float(conf[i]), int(cls[i])))
        return res


print("추론 엔진 :", ENGINE, "(XNNPACK 자동) / 스레드", THREADS)
det_standby = Detector(STANDBY_MODEL, "대기")
det_active = Detector(ACTIVE_MODEL, "감시")
det_hi = Detector(HI_MODEL, "경계·경보(고해상도)") if HI_MODEL else det_active
print("문턱값    : conf %.2f / iou %.2f   (--conf 로 바꿈)" % (CONF_TH, IOU_TH))
print("대기 모드 : %.0f초마다 한 장 추론 / 고성능 모드 유지 %.0f초 (--standby-interval / --hold)"
      % (STANDBY_INTERVAL, HOLD_SEC))
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
    near = None          # 가장 가까운 (아기, 위험물) 쌍의 거리 / 위험 반경. 없으면 None
    for dg in dangers:
        x1, y1, x2, y2, _, c = dg
        radius = (x2 - x1) * DANGER_SCALE
        circles.append((x1, y1, x2, y2, radius))
        for bb in babies:
            if DEPTH_CHECK > 0:
                _bh = bb[3] - bb[1]
                if _bh > 0 and (x2 - x1) / float(_bh) < DEPTH_CHECK:
                    n_depth_skip[0] += 1       # 위험물이 아기보다 훨씬 작게 보임 = 뒤쪽 = 이 쌍은 무시
                    continue
            gap = box_gap(bb[:4], dg[:4])
            if gap <= radius:
                hit = (NAMES[c], gap, radius)
            if radius > 0:
                ratio = gap / radius
                if near is None or ratio < near:
                    near = ratio
    return hit is not None, hit, circles, near


# ─────────────────────────────────────────────────────────────
#  카메라
# ─────────────────────────────────────────────────────────────
CAM_SOURCE = 0          # 영상 파일로 시험할 때는 여기를 파일 이름으로 바꾼다
cam_is_device = isinstance(CAM_SOURCE, int)
cam_reopen_t = []       # --cam-off-standby 로 카메라를 다시 여는 데 걸린 시간들
cam_released = [False]  # 지금 카메라를 놓아 둔 상태인가


def open_camera():
    c = cv2.VideoCapture(CAM_SOURCE)
    if cam_is_device:
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        c.set(cv2.CAP_PROP_FPS, 30)
        c.set(cv2.CAP_PROP_BUFFERSIZE, CAM_BUFFER)
    return c


cap = open_camera()
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


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except OSError:
        return None


# ── 모드 상태 (전력 스레드도 읽는다) ──────────────────────
mode = "standby"            # "standby"(대기) / "active"(감시) / "guard"(경계) / "alert"(경보)
MODE_KR = {"standby": "대기", "active": "감시", "guard": "경계", "alert": "경보"}
MODES = ("standby", "active", "guard", "alert")

# ── 전력 CSV ────────────────────────────────────────────────
#   1초마다 한 줄 : 시각, 경과(초), 전력(W), 그 순간의 모드, 누적 에너지(Wh), CPU 온도
#   한 줄씩 바로 flush 하므로 도중에 전원이 꺼져도 그때까지의 기록은 남는다.
_stamp = time.strftime("%Y%m%d_%H%M%S")
PW_LOG_CSV = (POWER_CSV + "_log.csv") if POWER_CSV else ("power_log_%s.csv" % _stamp)
PW_SUM_CSV = (POWER_CSV + "_summary.csv") if POWER_CSV else ("power_summary_%s.csv" % _stamp)
# 모드별 누적 : {모드: {"sec": 초, "wh": Wh, "n": 표본수, "min": W, "max": W}}
pw_stats = {m: {"sec": 0.0, "wh": 0.0, "n": 0, "min": None, "max": None} for m in MODES}
pw_total_wh = [0.0]
t_pw0 = time.time()


def power_worker(stop_flag, csv_path):
    """1초마다 전력을 읽어 CSV 에 쓰고 모드별로 쌓는다. 루프 타이밍을 건드리지 않는다.

    에너지(Wh) = 전력(W) x 직전 표본과의 실제 간격(초) / 3600.
    vcgencmd 한 번에 수십 ms 가 걸려 간격이 정확히 1초가 아니므로 실제 간격을 쓴다.
    """
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["timestamp", "elapsed_s", "power_w", "mode", "cum_energy_wh", "cpu_temp_c"])
        t_last = time.time()
        while not stop_flag[0]:
            w = read_power()
            now = time.time()
            dt = now - t_last
            t_last = now
            if w:
                m = mode                     # 이 표본이 속한 모드
                pw_log.append(w)
                st = pw_stats[m]
                st["sec"] += dt
                st["wh"] += w * dt / 3600.0
                st["n"] += 1
                st["min"] = w if st["min"] is None else min(st["min"], w)
                st["max"] = w if st["max"] is None else max(st["max"], w)
                pw_total_wh[0] += w * dt / 3600.0
                wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), round(now - t_pw0, 2),
                             round(w, 3), m, round(pw_total_wh[0], 6), cpu_temp()])
                f.flush()
            time.sleep(max(0.0, 1.0 - (time.time() - now)))


def write_power_summary(path, extra):
    """끝날 때 모드별 요약 CSV 를 쓴다. 시간당 Wh = 그 모드의 평균 W (1시간 계속 켰을 때)."""
    import csv
    tot_sec = sum(v["sec"] for v in pw_stats.values())
    rows = []
    for m in MODES:
        v = pw_stats[m]
        avg = (v["wh"] * 3600.0 / v["sec"]) if v["sec"] > 0 else None
        rows.append([m, MODE_KR[m], round(v["sec"], 1),
                     round(100.0 * v["sec"] / tot_sec, 1) if tot_sec else 0.0, v["n"],
                     round(avg, 3) if avg else "", round(v["min"], 3) if v["min"] else "",
                     round(v["max"], 3) if v["max"] else "", round(v["wh"], 6),
                     round(avg, 3) if avg else ""])
    tot_avg = (pw_total_wh[0] * 3600.0 / tot_sec) if tot_sec else None
    rows.append(["total", "전체", round(tot_sec, 1), 100.0 if tot_sec else 0.0,
                 sum(v["n"] for v in pw_stats.values()), round(tot_avg, 3) if tot_avg else "", "", "",
                 round(pw_total_wh[0], 6), round(tot_avg, 3) if tot_avg else ""])
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["mode", "mode_kr", "seconds", "time_share_pct", "samples",
                     "avg_power_w", "min_power_w", "max_power_w", "energy_wh", "wh_per_hour"])
        wr.writerows(rows)
        wr.writerow([])
        for k, v in extra.items():                  # 실행 조건도 함께 남긴다
            wr.writerow(["#", k, v])
    return rows


_pw_stop = [False]
PW_OK = read_power() is not None
if PW_OK:
    import threading
    threading.Thread(target=power_worker, args=(_pw_stop, PW_LOG_CSV), daemon=True).start()
    print("[전력] PMIC 에서 소비 전력을 1초마다 기록합니다 ->", PW_LOG_CSV)
else:
    print("[전력] vcgencmd 로 전력을 읽지 못했습니다 (라즈베리파이 5 에서만 됩니다). CSV 를 만들지 않습니다.")
try:
    import psutil
    psutil.cpu_percent(interval=None)
    psutil_ok = True
except ImportError:
    psutil_ok = False


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

# 두 모드 전환에 쓰는 값들
next_standby = 0.0      # 대기 모드에서 다음 추론 시각 (0 = 시작하자마자 한 장)
last_baby = 0.0         # 아기가 마지막으로 보인 시각 (감시 -> 대기 판단)
last_danger = 0.0       # 위험이 마지막으로 판정된 시각 (경보 -> 감시 판단)
n_to_active = 0         # 대기 -> 감시 전환 횟수
n_to_standby = 0        # 감시 -> 대기 전환 횟수
n_to_alert = 0          # 감시/경계 -> 경보 전환 횟수
n_alert_off = 0         # 경보 -> 경계 전환 횟수
n_to_guard = 0          # 감시 -> 경계 전환 횟수
n_guard_off = 0         # 경계 -> 감시 전환 횟수
last_near = 0.0         # 아기-위험물이 마지막으로 '가까웠던' 시각
mode_frames = {m: 0 for m in MODES}


def set_mode(new_mode, now, why):
    """모드를 바꾸고 이유를 찍는다. 전력 스레드는 다음 표본부터 새 모드로 기록한다."""
    global mode, next_standby, danger_since
    if new_mode == mode:
        return
    old = mode
    mode = new_mode
    if old == "alert" and new_mode != "alert":
        notify_streak[0] = 0          # 경보 구간이 끝나면 알림 간격을 30초부터 다시
    if CLOCK_LOW or CLOCK_HIGH:
        set_clock((CLOCK_HIGH or _clock_default) if new_mode in ("guard", "alert") else (CLOCK_LOW or _clock_default))
    if new_mode == "standby":
        next_standby = now + STANDBY_INTERVAL
        dwell_hist.clear()          # 감시·경보의 체류 기록은 넘기지 않는다
        danger_since = None
        set_led("green")
    print("[모드] %s -> %s  (%s)  경과 %.1f초"
          % (MODE_KR[old], MODE_KR[new_mode], why, now - t_run0))

try:
    while True:
        # ── 대기 모드 : 다음 추론 시각까지 쉰다 ─────────────
        #   CPU 가 쉬어야 전력이 준다. 화면 창이 있으면 q 키를 받을 수 있게 짧게 나눠 기다린다.
        if mode == "standby":
            with span("standby-wait"):
                if CAM_OFF_STANDBY and cam_is_device and cap.isOpened() \
                        and next_standby - time.time() > 1.0:
                    cap.release()            # 다음 추론까지 1초 넘게 남았으면 카메라를 놓는다
                    cam_released[0] = True
                quit_now = False
                while True:
                    remain = next_standby - time.time()
                    if remain <= 0 or (RUN_SEC and time.time() - t_run0 >= RUN_SEC):
                        break
                    if SHOW_GUI:
                        if cv2.waitKey(int(min(remain, 0.1) * 1000) or 1) & 0xFF == ord("q"):
                            quit_now = True
                            break
                    else:
                        time.sleep(min(remain, 0.1))
                if quit_now:
                    break
                if CAM_OFF_STANDBY and cam_is_device and cam_released[0]:
                    cam_released[0] = False
                    # 쉬는 동안 놓아 둔 카메라를 다시 연다 (걸린 시간을 기록)
                    _t = time.time()
                    cap = open_camera()
                    # 막 연 V4L2 장치는 처음 몇 장을 못 주거나 어둡게 준다. 1초 안에서 예열한다.
                    _ok = False
                    for _ in range(30):
                        _ok, _ = cap.read()
                        if _ok:
                            break
                        time.sleep(0.03)
                    cam_reopen_t.append(time.time() - _t)
                    if not cap.isOpened() or not _ok:
                        print("카메라를 다시 열 수 없습니다.")
                        break
                # 쉬는 동안 버퍼에 쌓인 낡은 프레임을 버린다 (몇 초 전 장면으로 판단하지 않게)
                for _ in range(CAM_BUFFER):
                    cap.grab()

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
            if GATE and (mode == "active" or (GUARD_GATE and mode == "guard")):
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

            mode_frames[mode] += 1
            if mode == "standby":
                # ── 대기 : SSDLite 로 한 장. 아기가 보이면 바로 감시 상태로 ──
                dets = det_standby(frame)
                next_standby = time.time() + STANDBY_INTERVAL
                if any(d[5] == SUBJECT for d in dets):
                    n_to_active += 1
                    set_mode("active", time.time(), "아기 감지")
                    last_baby = time.time()          # 전환 직후 HOLD_SEC 동안은 유지
                    dets = det_active(frame)         # 같은 장면을 정확한 모델로 다시 본다
                prev_dets = dets
            elif skip:
                n_skip += 1
                n_skip_run += 1
                dets = prev_dets          # 직전 결과를 그대로 쓴다. 비우지 않는다
            else:
                n_skip_run = 0
                # Detector 안에서 preproc / inference / postproc 구간이 기록된다
                # 경계·경보에서는 고해상도 모델(있으면), 감시에서는 가벼운 모델
                dets = (det_hi if mode in ("guard", "alert") else det_active)(frame)
                prev_dets = dets

            with span("judge"):
                is_danger, info, circles, near = judge(dets)
            prev_danger = is_danger     # 위험 중이면 다음 프레임은 절대 안 건너뛴다
            now = time.time()

            # ── 3상태 전이 (2026-09-26 수정) ──
            #   감시 -> 경보 : 위험 판정이 나는 즉시 (게이팅·속도 상한이 풀린다)
            #   경보 -> 감시 : 위험이 HOLD_SEC 동안 안 나면
            #   감시 -> 대기 : 아기가 HOLD_SEC 동안 안 보이면 (위험물 유무와 무관)
            #   예전 판은 "아기+위험물이 함께 안 보이면 대기" 라서 아기만 있을 때
            #   대기(5초) <-> 고성능(3초) 를 무한 왕복했다 (9/26 실측 4.21 W).
            # ── 4상태 전이 ──
            #   감시 -> 경계 : 아기-위험물 거리가 반경의 GUARD_SCALE 배 안 (가까워지는 중)
            #   경계 -> 감시 : 그 조건이 HOLD_SEC 동안 안 맞으면
            #   감시/경계 -> 경보 : 위험 반경 안 (즉시)
            #   경보 -> 경계 : 위험이 HOLD_SEC 동안 없으면
            #   감시/경계 -> 대기 : 아기가 HOLD_SEC 동안 안 보이면
            _near_ok = (near is not None and near < GUARD_SCALE)
            if mode in ("active", "guard", "alert") and any(d[5] == SUBJECT for d in dets):
                last_baby = now
            if _near_ok:
                last_near = now
            if mode == "active":
                if is_danger:
                    n_to_alert += 1
                    last_danger = now
                    set_mode("alert", now, "위험 판정")
                elif _near_ok:
                    n_to_guard += 1
                    set_mode("guard", now, "아기-위험물 거리가 반경의 %.1f배 안 (비율 %.2f)" % (GUARD_SCALE, near))
                elif now - last_baby >= HOLD_SEC:
                    n_to_standby += 1
                    set_mode("standby", now, "아기가 %.0f초 동안 안 보임" % HOLD_SEC)
            elif mode == "guard":
                if is_danger:
                    n_to_alert += 1
                    last_danger = now
                    set_mode("alert", now, "위험 판정")
                elif now - last_baby >= HOLD_SEC:
                    n_to_standby += 1
                    set_mode("standby", now, "아기가 %.0f초 동안 안 보임" % HOLD_SEC)
                elif now - last_near >= HOLD_SEC:
                    n_guard_off += 1
                    set_mode("active", now, "위험물과 멀어진 지 %.0f초" % HOLD_SEC)
            elif mode == "alert":
                if is_danger:
                    last_danger = now
                elif now - last_danger >= HOLD_SEC:
                    n_alert_off += 1
                    last_near = now
                    set_mode("guard", now, "위험이 %.0f초 동안 없음" % HOLD_SEC)

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
                # 같은 경보가 이어질 때 알림 간격 늘리기 (알림만 건너뛰고 LED·소리·웹은 그대로)
                _send = True
                if ALARM_BACKOFF and notify_streak[0] > 0:
                    _gap = NOTIFY_GAPS[min(notify_streak[0] - 1, len(NOTIFY_GAPS) - 1)]
                    if now - last_notify_t[0] < _gap:
                        _send = False
                        n_notify_skip[0] += 1
                        print("[알림] 같은 경보가 이어져 알림 건너뜀 (보낸 알림 %d회, 다음까지 %.0f초)"
                              % (notify_streak[0], _gap - (now - last_notify_t[0])))
                if _send:
                    notify_streak[0] += 1
                    last_notify_t[0] = now
                try:
                    if noti and _send:
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
                        # 4상태 판: 경계 구역(반경 x GUARD_SCALE)도 얇은 노란 원으로
                        cv2.circle(shot, ((bx1 + bx2) // 2, (by1 + by2) // 2),
                                   int(brad * GUARD_SCALE), (0, 220, 220), 1, cv2.LINE_AA)
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
                # 4상태 판: 웹 화면 아래 '모델' 칸에 지금 상태·속도 상한·클럭을 같이 보여 준다
                #   (web.py 를 안 고치고 보이게 하는 임시 방법. 화면의 FPS 숫자는 실제 루프 속도)
                try:
                    _cap = {"standby": "5초에 1장", "active": "%.0f FPS" % WATCH_FPS,
                            "guard": "%.0f FPS" % GUARD_FPS, "alert": ("%.0f FPS" % ALERT_FPS) if ALERT_FPS else "전속력"}[mode]
                    _mdl = os.path.basename(HI_MODEL if (HI_MODEL and mode in ("guard", "alert")) else (STANDBY_MODEL if mode == "standby" else ACTIVE_MODEL))
                    _pw = ("%.2f W" % pw_log[-1]) if pw_log else "측정 불가"
                    web._status["model"] = "%s | 상태: %s | 상한 %s | 클럭 %s | 전력(PMIC) %s" % (
                        _mdl, MODE_KR[mode], _cap, ((_clock_now[0] or "기본") if (CLOCK_LOW or CLOCK_HIGH) else "기본"), _pw)
                    web._status["power_w"] = round(pw_log[-1], 2) if pw_log else None
                except Exception:
                    pass

            # 화면
            if SHOW_GUI:
                with span("draw"):
                    for (x1, y1, x2, y2, rad) in circles:
                        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), int(rad),
                                   (0, 140, 255), 1, cv2.LINE_AA)
                        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), int(rad * GUARD_SCALE),
                                   (0, 220, 220), 1, cv2.LINE_AA)      # 경계 구역
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

            # ── 감시 상태의 속도 상한 + 조용할 때 쉬기 ────────────
            #   감시(아기는 있지만 위험 아님)에서는 루프를 WATCH_FPS 이상으로 돌리지 않고,
            #   게이팅으로 건너뛴 직후에는 GATE_IDLE_FPS 로 더 천천히 돈다.
            #   경보 상태와 대기 상태에서는 이 상한이 없다 (경보 = 전속력, 대기 = 5초 대기 루프).
            _cap_fps = 0.0
            if mode == "active":
                _cap_fps = WATCH_FPS
                if GATE and GATE_IDLE_FPS > 0 and n_skip_run > 0:
                    _cap_fps = min(GATE_IDLE_FPS, WATCH_FPS) if WATCH_FPS > 0 else GATE_IDLE_FPS
            elif mode == "guard":
                _cap_fps = GUARD_FPS
            elif mode == "alert":
                _cap_fps = ALERT_FPS
            if _cap_fps > 0:
                with span("idle"):
                    _target = 1.0 / _cap_fps
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
    if (CLOCK_LOW or CLOCK_HIGH) and _clock_default:
        set_clock(_clock_default)      # 원래 최대 클럭으로 되돌린다
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
    print(" 측정 요약   대기 %s / 고성능 %s"
          % (os.path.basename(STANDBY_MODEL), os.path.basename(ACTIVE_MODEL)))
    print("=" * 52)
    print("  돌린 시간      : %.1f 초" % total)
    print("  처리 프레임    : %d  (대기 %d / 고성능 %d)"
          % (n_frame, mode_frames["standby"], mode_frames["active"]))
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

    # ── 모드별 요약 (대기 / 고성능) + 요약 CSV ─────────────
    print()
    print("  ── 4상태 (대기 / 감시 / 경계 / 경보) ──")
    print("    대기   : %s  (%.0f초마다 1장, 추론 %d회, 프레임 %d)"
          % (os.path.basename(STANDBY_MODEL), STANDBY_INTERVAL, det_standby.n_infer, mode_frames["standby"]))
    print("    감시   : %s  (게이팅 %s, 상한 %.0f FPS, 프레임 %d)"
          % (os.path.basename(ACTIVE_MODEL), "켬" if GATE else "끔", WATCH_FPS, mode_frames["active"]))
    print("    경계   : %s  (상한 %.0f FPS, 반경 x%.1f 안, 게이팅 %s, 프레임 %d)"
          % (os.path.basename(HI_MODEL or ACTIVE_MODEL), GUARD_FPS, GUARD_SCALE, "켬" if (GATE and GUARD_GATE) else "끔", mode_frames["guard"]))
    print("    경보   : %s  (상한 %s, 게이팅 없음, 프레임 %d)"
          % (os.path.basename(HI_MODEL or ACTIVE_MODEL), ("%.0f FPS" % ALERT_FPS) if ALERT_FPS else "없음(전속력)", mode_frames["alert"]))
    _dets = [det_standby, det_active] + ([det_hi] if HI_MODEL else [])
    for _d in _dets:
        if _d.n_infer:
            print("      %s 추론 %d회, 평균 %.1f ms" % (_d.name, _d.n_infer, 1000.0 * _d.t_infer / _d.n_infer))
    print("    전환 횟수   : 대기->감시 %d / 감시->대기 %d / 감시->경계 %d / 경계->감시 %d / ->경보 %d / 경보->경계 %d"
          % (n_to_active, n_to_standby, n_to_guard, n_guard_off, n_to_alert, n_alert_off))
    if CLOCK_LOW or CLOCK_HIGH:
        print("    클럭 전환   : %d회 (대기·감시 %s / 경계·경보 %s kHz)" % (n_clock_sw[0], CLOCK_LOW or _clock_default, CLOCK_HIGH or _clock_default))
    if ALARM_BACKOFF:
        print("    알림 간격 늘리기 : 건너뛴 알림 %d회 (간격 %s초)" % (n_notify_skip[0], ",".join("%.0f" % g for g in NOTIFY_GAPS)))
    if DEPTH_CHECK > 0:
        print("    거리 비율 검사   : 위험물 폭/아기 높이 < %.2f 라 무시한 쌍 %d회" % (DEPTH_CHECK, n_depth_skip[0]))
    if PW_OK:
        _rows = write_power_summary(PW_SUM_CSV, {
            "engine": ENGINE, "threads": THREADS,
            "standby_model": os.path.basename(STANDBY_MODEL),
            "active_model": os.path.basename(ACTIVE_MODEL),
            "standby_interval_s": STANDBY_INTERVAL, "hold_s": HOLD_SEC, "conf_th": CONF_TH,
            "to_active_count": n_to_active, "to_standby_count": n_to_standby,
            "to_alert_count": n_to_alert, "alert_off_count": n_alert_off,
            "gate": GATE, "watch_fps": WATCH_FPS, "alert_fps": ALERT_FPS,
            "guard_fps": GUARD_FPS, "guard_scale": GUARD_SCALE, "guard_gate": GUARD_GATE, "hi_model": os.path.basename(HI_MODEL) if HI_MODEL else "",
            "clock_low": CLOCK_LOW, "clock_high": CLOCK_HIGH, "clock_switches": n_clock_sw[0],
            "threads": THREADS, "cam_off_standby": CAM_OFF_STANDBY, "idle_w": IDLE_W,
            "alarm_count": n_alarm, "run_seconds": round(total, 1)})
        print()
        print("    %-6s %8s %7s %9s %10s %10s" % ("모드", "시간(초)", "비율", "평균 W", "에너지 Wh", "시간당 Wh"))
        for r in _rows:
            print("    %-6s %8s %6s%% %9s %10s %10s" % (r[1], r[2], r[3], r[5] or "-", r[8], r[9] or "-"))
        print("    ※ 시간당 Wh = 그 모드로 1시간 계속 켰을 때 쓰는 에너지 (= 평균 W)")
        print("    CSV : %s (1초마다) / %s (요약)" % (PW_LOG_CSV, PW_SUM_CSV))
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
    # ── 단계별 시간과 에너지 추정 (저전력 실험 판) ───────────────
    #   에너지는 직접 못 잰다(PMIC 는 1초에 한 번, 단계는 ms). 그래서 이렇게 나눈다.
    #     동적 전력 = 평균 전력 - 유휴 전력(--idle-w)
    #     CPU 를 쓰는 단계에 그 동적 전력을 '시간 x 쓰는 코어 수' 비율로 배분
    #     (추론은 4스레드 = 코어 4개, 나머지 단계는 1개로 본다 -> 추정치다)
    if STAGE_T:
        wall = max(1e-6, total)
        cores = {"inference": max(1, THREADS)}
        busy = {k: v for k, v in STAGE_T.items() if k not in ("frame", "standby-wait", "idle")}
        weight = {k: v * cores.get(k, 1) for k, v in busy.items()}
        wsum = sum(weight.values()) or 1.0
        p_avg = (sum(pw_log) / len(pw_log)) if pw_log else None
        p_dyn = (p_avg - IDLE_W) if p_avg is not None else None
        if p_dyn is not None and p_dyn < 0.05:
            print()
            print("  ── 단계별 에너지 추정 생략: 평균 %.2f W 가 유휴 기준 %.2f W 이하 (대기만 한 실행)" % (p_avg, IDLE_W))
            p_dyn = None
        print()
        print("  ── 단계별 시간 · 에너지 추정 (유휴 %.2f W 기준) ──" % IDLE_W)
        print("    %-14s %7s %9s %8s %9s %10s" % ("단계", "횟수", "1회 ms", "시간%", "에너지%", "에너지 J"))
        for k, v in sorted(busy.items(), key=lambda kv: -weight[kv[0]]):
            n = STAGE_N.get(k, 0)
            e_share = 100.0 * weight[k] / wsum
            e_j = (p_dyn * wall * weight[k] / wsum) if p_dyn is not None else float("nan")
            print("    %-14s %7d %9.2f %7.1f%% %8.1f%% %10.2f" % (k, n, 1000.0 * v / max(1, n), 100.0 * v / wall, e_share, e_j))
        rest = wall - sum(busy.values())
        print("    %-14s %7s %9s %7.1f%% %8s %10s" % ("쉼/대기", "", "", 100.0 * rest / wall, "-", "-"))
        if p_dyn is not None and STAGE_N.get("inference"):
            n_inf = STAGE_N["inference"]
            e_inf = p_dyn * wall * weight.get("inference", 0.0) / wsum
            print("    추론 1회당 에너지 : 약 %.0f mJ  (동적 %.2f W x %.1f ms x 코어 비중)" % (1000.0 * e_inf / n_inf, p_dyn, 1000.0 * STAGE_T["inference"] / n_inf))
            print("    ※ 유휴 %.2f W 는 %.1f초 동안 %.1f J. 프로그램이 쓴 몫(동적)은 %.1f J." % (IDLE_W, wall, IDLE_W * wall, p_dyn * wall))
    if cam_reopen_t:
        print("  카메라 재개방  : %d회, 평균 %.2f초, 최대 %.2f초 (--cam-off-standby)"
              % (len(cam_reopen_t), sum(cam_reopen_t) / len(cam_reopen_t), max(cam_reopen_t)))
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
