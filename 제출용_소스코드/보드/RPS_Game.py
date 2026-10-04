# -*- coding: utf-8 -*-
"""
가위바위보 게임 (2인용) — 프로젝트 본체

강사님 공지의 5단계를 구현한다.
  1단계 좌/우 플레이어 구분 : Bounding Box X좌표로 P1(왼쪽), P2(오른쪽)
  2단계 승패 로직          : P1 Class vs P2 Class 비교 → 승/무/패
  3단계 예외 처리          : 손이 2개가 아닐 때 안내 메시지
  4단계 타이밍 제어        : 3-2-1 카운트다운 후 판정 순간 프레임 고정(Freeze)
  5단계 게임성 확장        : 스코어보드, 효과음, 재시작 키

바탕: EX_03_Board_RPS_PreTrained_YOLO.py
  - 검출 부분(letterbox, 전처리, NMS, 좌표 복원)은 그대로 가져왔다.
  - 04 실습에서 효과를 확인한 개선(waitKey=1, 카메라 스레드, num_threads)을 넣었다.
  - 검출과 게임 로직을 함수로 나눠, 게임 규칙만 따로 읽을 수 있게 했다.

실행:  python3 RPS_Game.py
       python3 RPS_Game.py my_best.tflite      <- 모델을 바꿔 가며 시험

조작:  SPACE  한 판 시작 (3-2-1 카운트다운)
       r      점수 초기화
       q      종료
"""

# ─────────────────────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL  = 'my_best_int8.tflite'
CONF_TH        = 0.10    # 신뢰도 문턱값
#   0.4 -> 0.10 으로 낮췄다. 시험지 120장으로 실측한 결과다(2026-09-18).
#     0.40 : 판정 정확도 90.3% / 판정을 내린 자세 25.8%
#     0.10 : 판정 정확도 78.4% / 판정을 내린 자세 80.8%
#   0.35~0.10 구간에서 정확도는 78~81%로 거의 평평한데 판정 비율만 2.6배 오른다.
#   0.05 로 더 내리면 정확도가 72.4%로 꺾이므로 0.10 이 최적점이다.
#   손이 없는 배경 사진 15장에서는 어느 값에서도 오검출이 0개였다.
IOU_TH         = 0.45    # NMS 겹침 문턱값
IMG_SIZE       = 320     # 모델 입력 크기 (학습 때 imgsz와 같아야 한다)

WAIT_MS        = 1       # cv2.waitKey 대기(ms)
CAMERA_THREAD  = True    # 카메라를 별도 스레드에서 미리 읽기
INTERP_THREADS = 4       # 추론 스레드 수 (라즈베리파이 5는 코어 4개)

COUNTDOWN_SEC  = 1.0     # 숫자 하나당 유지 시간 (3-2-1 이므로 총 3초)
RESULT_SEC     = 2.5     # 결과 화면을 보여 주는 시간
GRACE_SEC      = 2.0     # 카운트 0 이후, 손 2개를 기다려 주는 시간
NOTICE_SEC     = 2.0     # 취소 등 안내 문구를 띄워 두는 시간
# 효과음 (5단계 '승패 효과음').
#   True 로 두면 판정할 때 소리가 난다. 나는 방식은 두 가지고 자동으로 고른다.
#     (1) win/lose/draw.wav 가 옆에 있고 aplay 가 되면 -> 그 소리를 낸다
#         wav 는 make_sounds.py 로 만든다. 라즈베리파이 5 는 3.5mm 잭이 없으므로
#         HDMI 스피커·USB 스피커·블루투스 중 하나가 있어야 들린다.
#     (2) 그렇지 않으면 -> 터미널 벨. SSH 를 타고 넘어와 MobaXterm 이 켜진
#         PC 스피커에서 울린다. 보드에서 나는 소리가 아니라는 점은
#         보고서에 그대로 밝히는 편이 낫다.
SOUND          = True
MIRROR         = True    # 화면을 거울처럼 좌우 반전. 사람이 보고 조작하기 편하다

# 얼굴이 손으로 오인식될 때 쓰는 임시 방편.
#   True면 박스가 3개 이상일 때 '신뢰도 상위 2개'만 손으로 본다.
#   근본 해결은 얼굴이 함께 찍힌 사진을 배경(background)으로 학습시키는 것이다.
#   임시 방편을 켜면 3단계 예외 처리(손 3개 안내)가 화면에 안 나타나므로,
#   과제 시연 때는 False로 두고 카메라 각도로 얼굴을 빼는 편이 낫다.
USE_TOP2       = True
#   CONF_TH 를 0.10 으로 낮추면 박스가 3개 이상 나오는 자세가 생긴다.
#   이 값이 False 면 그때 게임이 'Too many hands' 로 판정을 거부한다.
#   위 실측은 '신뢰도 높은 2개'를 고른 기준이므로 같이 켜야 그 수치가 나온다.

# 웹캠이 영상을 보내는 형식(FOURCC, 네 글자 코드).
#   'MJPG' = Motion-JPEG. 카메라가 한 장씩 JPEG으로 압축해서 USB로 보낸다.
#   'YUYV' = 무압축. 화질 손실은 없지만 USB 대역폭을 크게 먹어 장수가 줄어든다.
#   OpenCV 기본값은 보통 YUYV라, MJPG로 바꾸면 같은 USB로 더 많은 장수를 받는다.
#   'NONE'으로 두면 아무것도 건드리지 않는다.
FOURCC         = 'MJPG'
CAM_FPS        = 30      # 카메라에 요청할 프레임률. 0이면 요청하지 않음
# ─────────────────────────────────────────────────────────────

import os
import subprocess
import sys
import time
import threading

import numpy as np
import cv2
# LiteRT 인터프리터를 가져온다.
# 같은 물건인데 패키지 이름이 바뀌었다.
#   옛 이름 : tflite_runtime.interpreter   (구글이 TensorFlow Lite로 부르던 시절)
#   새 이름 : ai_edge_litert.interpreter   (LiteRT로 이름이 바뀐 뒤)
# 강사님 공지(2026-09-17)로 05 예제는 새 이름을 쓰기로 했다.
# 보드에 둘 중 무엇이 깔려 있든 돌아가도록 새 이름을 먼저 시도한다.
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

modelPath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
print('model path:', modelPath)

# ── 모델 로딩 ────────────────────────────────────────────────
try:
    interpreter = tflite.Interpreter(model_path=modelPath, num_threads=INTERP_THREADS)
except TypeError:
    interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

CLASS_NAME = {0: 'scissors', 1: 'rock', 2: 'paper'}
CLASS_KR = {0: '가위', 1: '바위', 2: '보'}   # 터미널 출력용. 화면에는 못 쓴다(한글 미지원)
CLASS_COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]   # BGR: 가위 파랑 / 바위 초록 / 보 빨강

# 화면에 쓰는 색
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


# ─────────────────────────────────────────────────────────────
#  카메라 (별도 스레드에서 미리 읽어 두면 메인 루프가 안 기다린다)
# ─────────────────────────────────────────────────────────────
class CameraThread:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_flag.is_set():
            ok, f = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, f
            if not ok:
                break

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self):
        self.stop_flag.set()
        self.thread.join(timeout=1.0)


class DirectCamera:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        return self.cap.read()

    def release(self):
        pass


# ─────────────────────────────────────────────────────────────
#  검출 (EX_03에서 가져온 부분)
# ─────────────────────────────────────────────────────────────
def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    """비율을 유지한 채 정사각형으로. 남는 자리는 회색으로 채운다."""
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_x = (nw - new_w) // 2
    pad_y = (nh - new_h) // 2

    padded = cv2.copyMakeBorder(resized, pad_y, nh - new_h - pad_y,
                                pad_x, nw - new_w - pad_x,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, pad_x, pad_y


def detect(frame):
    """한 프레임에서 손을 찾아 목록으로 돌려준다.

    반환: [(x1, y1, x2, y2, cls, score), ...]   좌표는 원본 frame 기준
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lb, r, pad_x, pad_y = letterbox(img_rgb)

    img = img_lb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)          # (320,320,3) -> (1,320,320,3)
    img = np.transpose(img, (0, 3, 1, 2))      # -> (1,3,320,320)  NCHW

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_index)[0].transpose()   # (2100, 7)

    class_scores = raw[:, 4:]
    confidences = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    keep_mask = confidences > CONF_TH
    filtered = raw[keep_mask]
    scores = confidences[keep_mask]
    classes = class_ids[keep_mask]

    if len(scores) == 0:
        return []

    cx, cy, bw, bh = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=-1)

    keep = cv2.dnn.NMSBoxesBatched(boxes, scores, classes,
                                   score_threshold=CONF_TH, nms_threshold=IOU_TH)

    result = []
    for i in keep:
        x, y, bw_, bh_ = boxes[i]
        x *= IMG_SIZE; y *= IMG_SIZE; bw_ *= IMG_SIZE; bh_ *= IMG_SIZE
        x1 = int(np.clip((x - pad_x) / r, 0, frame.shape[1]))
        y1 = int(np.clip((y - pad_y) / r, 0, frame.shape[0]))
        x2 = int(np.clip((x + bw_ - pad_x) / r, 0, frame.shape[1]))
        y2 = int(np.clip((y + bh_ - pad_y) / r, 0, frame.shape[0]))
        result.append((x1, y1, x2, y2, int(classes[i]), float(scores[i])))
    return result


# ─────────────────────────────────────────────────────────────
#  1단계 — 좌/우 플레이어 구분
# ─────────────────────────────────────────────────────────────
def split_players(boxes):
    """박스 2개를 X좌표 기준으로 P1(왼쪽), P2(오른쪽)로 나눈다.

    박스의 '중심 X'로 비교한다. 좌상단 x로 비교하면 손 크기가 다를 때
    가까이 붙은 두 손의 앞뒤가 뒤바뀔 수 있다.

    반환: (P1, P2)  각각 (x1, y1, x2, y2, cls, score)
    """
    def center_x(b):
        return (b[0] + b[2]) / 2

    ordered = sorted(boxes, key=center_x)
    return ordered[0], ordered[1]


# ─────────────────────────────────────────────────────────────
#  2단계 — 승패 로직
# ─────────────────────────────────────────────────────────────
# 0=가위, 1=바위, 2=보
# 가위는 보를, 바위는 가위를, 보는 바위를 이긴다.
BEATS = {0: 2, 1: 0, 2: 1}


def judge(p1_cls, p2_cls):
    """P1 기준 결과를 돌려준다.  'draw' | 'p1' | 'p2'"""
    if p1_cls == p2_cls:
        return 'draw'
    if BEATS[p1_cls] == p2_cls:
        return 'p1'
    return 'p2'


def result_text(who):
    return {'p1': 'P1 WIN', 'p2': 'P2 WIN', 'draw': 'DRAW'}[who]


def result_color(who):
    return {'p1': GREEN, 'p2': RED, 'draw': YELLOW}[who]


# ─────────────────────────────────────────────────────────────
#  4단계 — 게임 진행 상태
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  효과음
# ─────────────────────────────────────────────────────────────
def _find_aplay():
    """aplay 가 있고 출력 장치가 하나라도 있으면 True."""
    try:
        r = subprocess.run(['aplay', '-l'], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=2)
        return r.returncode == 0 and b'card' in r.stdout
    except Exception:
        return False


HAS_APLAY = _find_aplay() if SOUND else False
if SOUND:
    if HAS_APLAY:
        print('[sound] aplay 사용 (win/lose/draw.wav)')
    else:
        print('[sound] 출력 장치가 없어 터미널 벨을 쓴다 (PC 스피커에서 울린다)')


def play(kind):
    """kind: 'win' | 'draw'. 소리 때문에 게임이 멈추면 안 되므로 기다리지 않는다."""
    if not SOUND:
        return
    wav = kind + '.wav'
    if HAS_APLAY and os.path.exists(wav):
        try:
            subprocess.Popen(['aplay', '-q', wav],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    print('\a', end='', flush=True)      # 예비: 터미널 벨


class Game:
    """상태 기계.

        READY ─ SPACE ─▶ COUNT ─ 3초 ─▶ JUDGING ─ 손 2개 ─▶ RESULT ─▶ READY
                                          └ 2초 안에 못 찾으면 취소 ─▶ READY

    JUDGING(유예 구간)을 둔 이유:
      카운트가 0이 되는 그 '한 프레임'에 손이 정확히 2개로 잡혀야만 판정하면,
      11 FPS 환경에서 거의 매번 실패한다. 손을 내미는 동작 중에는 검출이
      흔들리기 때문이다. 그래서 0이 된 뒤 잠깐 기다렸다가
      '손 2개가 처음 잡히는 프레임'으로 판정한다.
    """

    READY = 'READY'
    COUNT = 'COUNT'
    JUDGING = 'JUDGING'
    RESULT = 'RESULT'

    def __init__(self):
        self.state = Game.READY
        self.t0 = 0.0             # 현재 상태가 시작된 시각
        self.score = {'p1': 0, 'p2': 0, 'draw': 0}
        self.frozen = None        # 판정 순간에 고정한 화면
        self.verdict = None       # ('p1'|'p2'|'draw', p1, p2)
        self.notice = ''          # 잠깐 띄우는 안내 (취소 등)
        self.notice_t = 0.0

    def say(self, text):
        self.notice = text
        self.notice_t = time.time()

    def notice_alive(self):
        return self.notice and time.time() - self.notice_t < NOTICE_SEC

    def start(self):
        if self.state == Game.READY:
            self.state = Game.COUNT
            self.t0 = time.time()
            self.notice = ''

    def reset_score(self):
        self.score = {'p1': 0, 'p2': 0, 'draw': 0}
        self.say('Score reset')

    def countdown_number(self):
        """3 → 2 → 1. 시간이 다 되면 0."""
        left = COUNTDOWN_SEC * 3 - (time.time() - self.t0)
        if left <= 0:
            return 0
        return min(3, int(left / COUNTDOWN_SEC) + 1)

    def finish(self, frame, boxes):
        """손 2개가 잡힌 순간 호출. 이 프레임을 고정하고 판정한다."""
        p1, p2 = split_players(boxes)
        who = judge(p1[4], p2[4])
        self.score[who] += 1
        self.verdict = (who, p1, p2)
        self.frozen = frame.copy()          # ← 4단계 Freeze
        self.state = Game.RESULT
        self.t0 = time.time()
        # 2인용이라 한쪽의 승은 곧 다른 쪽의 패다. 무승부만 따로 구분한다
        play('draw' if who == 'draw' else 'win')

    def maybe_end_result(self):
        if self.state == Game.RESULT and time.time() - self.t0 > RESULT_SEC:
            self.state = Game.READY
            self.frozen = None
            self.verdict = None


# ─────────────────────────────────────────────────────────────
#  화면 그리기
# ─────────────────────────────────────────────────────────────
def draw_boxes(frame, boxes, labels=None):
    """검출 박스와 이름을 그린다. labels를 주면 그 글자를 대신 쓴다."""
    for i, (x1, y1, x2, y2, cls, sc) in enumerate(boxes):
        color = CLASS_COLOR[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = labels[i] if labels else '%s %d%%' % (CLASS_NAME[cls], int(sc * 100))
        cv2.putText(frame, text, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_PLAIN, 1, color, 2)


def draw_center(frame, text, color, scale=3, thick=3, dy=0):
    """화면 한가운데에 큰 글자를 쓴다."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (frame.shape[1] - tw) // 2
    y = (frame.shape[0] + th) // 2 + dy
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_scoreboard(frame, game):
    """5단계 — 왼쪽 위 스코어보드."""
    s = game.score
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(frame, 'P1 %d  DRAW %d  P2 %d' % (s['p1'], s['draw'], s['p2']),
                (5, 13), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)


def draw_hint(frame, text, color=WHITE):
    """아래쪽 안내 문구."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 18), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, text, (5, h - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)


# ─────────────────────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
# 형식(FOURCC)은 해상도보다 먼저 지정해야 드라이버가 제대로 받아들인다
if FOURCC != 'NONE':
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if CAM_FPS:
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 요청한 대로 됐는지 확인해서 알려 준다 (set은 '부탁'이지 '명령'이 아니다)
_cc = int(cap.get(cv2.CAP_PROP_FOURCC))
print('[camera] %dx%d  형식=%s  FPS=%.0f'
      % (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
         ''.join(chr((_cc >> (8 * i)) & 0xFF) for i in range(4)),
         cap.get(cv2.CAP_PROP_FPS)))
source = CameraThread(cap) if CAMERA_THREAD else DirectCamera(cap)

cv2.namedWindow('RPS', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RPS', 320 * 2, 240 * 2)

game = Game()
fps = 0.0
prev = time.time()

print('=' * 52)
print(' SPACE 한 판 시작 | r 점수 초기화 | q 종료')
print('=' * 52)

try:
    while True:
        ok, frame = source.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        # 좌우 반전(거울 모드).
        #   웹캠 원본은 '카메라가 본 그대로'라 오른손이 화면 왼쪽에 나온다.
        #   거울처럼 뒤집으면 내 오른손이 화면 오른쪽에 나와 직관적이다.
        #   반드시 detect() '앞'에서 뒤집어야 박스 좌표까지 같이 맞는다.
        if MIRROR:
            frame = cv2.flip(frame, 1)      # 1 = 좌우, 0 = 상하, -1 = 둘 다

        boxes = detect(frame)

        # 임시 방편: 3개 이상 잡히면 신뢰도가 높은 2개만 남긴다
        # 추려 내기 전의 개수를 남겨 둔다.
        #   USE_TOP2 로 2개만 남기더라도 "몇 개가 보였는지"는 알려 줘야 한다.
        #   그렇지 않으면 3단계 예외 처리(손이 2개가 아닐 때 안내)가 화면에서 사라진다.
        n_found = len(boxes)
        if USE_TOP2 and n_found > 2:
            boxes = sorted(boxes, key=lambda b: -b[5])[:2]

        # ── 3단계 예외 처리 : 지금 화면의 상태를 문구로 만든다 ──
        if n_found < 2:
            status = 'Need 2 hands (found %d)' % n_found
            status_color = RED
        elif n_found > 2:
            # 추려서 판정은 계속하되, 손이 더 보였다는 사실은 그대로 알린다
            status = 'Too many hands (%d). Using top 2' % n_found
            status_color = RED
        else:
            status = 'Ready - 2 hands detected'
            status_color = GREEN

        # ── 4단계 타이밍 제어 ───────────────────────────────
        if game.state == Game.COUNT:
            if game.countdown_number() == 0:
                game.state = Game.JUDGING       # 유예 구간으로
                game.t0 = time.time()

        elif game.state == Game.JUDGING:
            if len(boxes) == 2:
                game.finish(frame, boxes)       # 판정 + Freeze
            elif time.time() - game.t0 > GRACE_SEC:
                game.state = Game.READY
                game.say('Cancelled: could not see 2 hands')

        game.maybe_end_result()

        # ── 화면 만들기 ─────────────────────────────────────
        if game.state == Game.RESULT and game.frozen is not None:
            # 판정 순간에 고정한 화면을 그대로 보여 준다
            view = game.frozen.copy()
            who, p1, p2 = game.verdict
            draw_boxes(view, [p1, p2],
                       labels=['P1 ' + CLASS_NAME[p1[4]], 'P2 ' + CLASS_NAME[p2[4]]])
            draw_center(view, result_text(who), result_color(who), scale=1.2, thick=3)
            # cv2.putText 는 한글을 못 그린다(??? 로 찍힌다). 화면에는 영어 이름을 쓴다.
            draw_hint(view, 'P1 %s  vs  P2 %s' % (CLASS[p1[4]], CLASS[p2[4]]))
        else:
            view = frame
            if len(boxes) == 2:
                p1, p2 = split_players(boxes)
                draw_boxes(view, [p1, p2],
                           labels=['P1 ' + CLASS_NAME[p1[4]], 'P2 ' + CLASS_NAME[p2[4]]])
            else:
                draw_boxes(view, boxes)

            if game.state == Game.COUNT:
                n = game.countdown_number()
                if n > 0:
                    draw_center(view, str(n), YELLOW, scale=3, thick=4)
                # 카운트 중에도 손이 몇 개 잡히는지 알려 준다
                draw_hint(view, 'Get ready...  ' + status, status_color)

            elif game.state == Game.JUDGING:
                draw_center(view, 'GO!', YELLOW, scale=2, thick=4)
                left = GRACE_SEC - (time.time() - game.t0)
                draw_hint(view, 'Show your hands!  (%.1fs)  %s' % (left, status),
                          status_color)

            elif game.notice_alive():
                draw_hint(view, game.notice, YELLOW)

            elif len(boxes) == 2:
                draw_hint(view, 'Press SPACE to play', GREEN)
            else:
                draw_hint(view, status, status_color)

        draw_scoreboard(view, game)

        now = time.time()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(view, 'FPS %.1f' % fps, (view.shape[1] - 70, 13),
                    cv2.FONT_HERSHEY_PLAIN, 1, YELLOW, 1)

        cv2.imshow('RPS', view)

        key = cv2.waitKey(WAIT_MS) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            game.start()
        elif key == ord('r'):
            game.reset_score()

except KeyboardInterrupt:
    pass

s = game.score
print()
print('=' * 52)
print(' 최종 점수   P1 %d   무승부 %d   P2 %d' % (s['p1'], s['draw'], s['p2']))
print('=' * 52)

source.release()
cap.release()
cv2.destroyAllWindows()
