# -*- coding: utf-8 -*-
"""
자동 촬영기 — EX_01_Image_Capture.py를 학습 데이터 모으기 편하게 고친 것.

강사님 원본은 한 장 찍을 때마다 's'를 눌러야 한다. 60장을 모으려면 60번 눌러야 하고,
손은 카메라 앞에 있어야 하니 혼자서는 거의 불가능하다.

이 파일이 하는 일
  1) INTERVAL초마다 자동으로 한 장씩 찍는다 (화면에 남은 시간이 보인다)
  2) 직전에 저장한 사진과 너무 비슷하면 건너뛴다
     → 같은 자세로 가만히 있으면 똑같은 사진 30장이 쌓이는 것을 막는다.
        '많이'가 아니라 '다양하게' 모아야 하기 때문이다.
  3) 파일 이름에 무엇을 찍는 중인지 남긴다 → 나중에 LabelImg로 라벨링할 때 훨씬 빠르다
  4) 이미 있는 파일 번호 뒤에서 이어 붙인다 → 강사님 사진(img_0001.jpg…)을 덮어쓰지 않는다

    ⚠ 강사님 원본은 항상 img_0001.jpg부터 저장한다. 데이터셋 폴더에서 그대로 돌리면
       기존 사진을 덮어쓴다. 이 파일은 그 사고가 안 나게 되어 있다.

조작
    1 / 2 / 3   찍는 대상 바꾸기 : 가위 / 바위 / 보
    4           손 2개 (섞인 것)
    0           손 없음 (배경·얼굴만) ← 얼굴 오검출 고치는 데 꼭 필요하다
    SPACE       자동 촬영 멈춤 / 다시 시작
    s           지금 즉시 한 장 (수동)
    q           끝내기

실행
    cd ~/work/examples/05_Object_Detection_Based_On-Device_AI
    python3 EX_01_Auto_Capture.py
"""
import os
import time
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────────────────────
SAVE_DIR   = './captures'   # 저장 폴더. 없으면 만든다
TAG        = 'brown'        # 이번 촬영을 한마디로. 예: brown(밤색 티), dark(어두운 곳), wall(흰 벽)
INTERVAL   = 1.0            # 몇 초마다 한 장

# 직전 저장본과 얼마나 달라야 저장할지 (평균 밝기 차이, 0~255)
#   0    : 검사 안 함. 무조건 다 찍는다
#   3.0  : 카메라가 완전히 멈춰 있을 때 생기는 '진짜 중복'만 버린다  ← 권장
#   8.0  : 자세가 제법 바뀌어야 저장한다 (장수가 확 줄어든다)
#
# 왜 0으로 두지 않는가 — 모델 성능 때문이 아니라 두 가지 실제 비용 때문이다.
#   ① 라벨링: 한 장마다 LabelImg로 손수 박스를 그려야 한다. 0.5초 무제한이면
#      1분에 120장이 쌓이는데, 그중 상당수가 사실상 같은 그림이다.
#   ② 시험지 오염: 거의 같은 사진 두 장이 train과 test에 나뉘어 들어가면
#      모델이 이미 본 것을 시험 보는 셈이라 mAP가 실제보다 높게 나온다.
#      점수를 못 믿게 되는 것이 진짜 손해다.
#
# 반대로 "비슷한 자세를 버리면 모델이 미세한 차이를 못 배우는 것 아니냐"는 걱정은
# 하지 않아도 된다. 건너뛴 프레임은 '잘못 가르치는' 것이 아니라 '안 가르치는' 것이다.
# 조금씩 기울어진 자세에 대한 내성은 학습 명령의 degrees=15.0 증강이 만들어 준다.
DIFF_TH    = 3.0
WIDTH      = 640            # 기존 데이터셋이 640x480이다. 맞춰야 한다
HEIGHT     = 480
FOURCC     = 'MJPG'
MIRROR     = True           # 화면만 좌우 반전(거울). 저장되는 사진은 원본 그대로

# 화면 한가운데에 세로 안내선을 그린다 (저장되는 사진에는 안 들어간다).
# 게임이 상자의 x 중심으로 P1/P2를 가르므로, 두 손을 이 선 좌우로 하나씩 두고 찍는다.
GUIDE      = True

SUBS = {ord('1'): 'scissors',
        ord('2'): 'rock',
        ord('3'): 'paper',
        ord('4'): 'two',      # 손 2개 (서로 다른 모양)
        ord('0'): 'bg'}       # 손 없음 = 배경 사진

# ─────────────────────────────────────────────────────────────
#  준비
# ─────────────────────────────────────────────────────────────
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print('[폴더 생성] %s' % os.path.abspath(SAVE_DIR))


def next_index():
    """이미 있는 사진 뒤 번호부터 시작한다. 덮어쓰기 사고 방지."""
    n = 0
    for f in os.listdir(SAVE_DIR):
        if not f.lower().endswith('.jpg'):
            continue
        base = os.path.splitext(f)[0]
        tail = base.split('_')[-1]
        if tail.isdigit():
            n = max(n, int(tail))
    return n + 1


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 카메라 예열 — 자동노출이 잡힐 때까지 몇 장 버린다 (첫 장이 어둡게 나오는 것 방지)
for _ in range(15):
    cap.read()

cv2.namedWindow('capture', cv2.WINDOW_NORMAL)
cv2.resizeWindow('capture', WIDTH, HEIGHT)

cnt = next_index()
n_skip = 0              # 너무 비슷해서 건너뛴 횟수
sub = 'two'
auto = False            # 멈춘 상태로 시작. 자세를 잡고 SPACE 를 눌러야 찍기 시작한다
t_next = time.time() + INTERVAL
last_saved = None       # 직전에 저장한 사진 (작게 줄인 것)
saved_by_sub = {}
msg, msg_t = '', 0.0

print('=' * 62)
print(' 저장 위치 : %s' % os.path.abspath(SAVE_DIR))
print(' 시작 번호 : %04d   (기존 파일 뒤에서 이어 붙인다)' % cnt)
print(' 1 가위   2 바위   3 보   4 손2개   0 손없음(배경)')
print('   -> 이 키를 누르면 그 대상으로 %.1f초마다 자동 촬영 시작' % INTERVAL)
print(' SPACE 멈춤/재개    s 수동 한 장    q 종료')
print('')
print(' >> 멈춘 상태로 시작합니다. 자세를 잡고 창을 클릭한 뒤 숫자 키를 누르세요. <<')
print('=' * 62)


def small(img):
    """비슷한지 비교할 때 쓰는 축소판. 작게 줄이면 잡음에 덜 흔들린다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 48)).astype(np.float32)


def save(frame):
    global cnt, last_saved, msg, msg_t
    name = '%s_%s_%04d.jpg' % (TAG, sub, cnt)
    cv2.imwrite(os.path.join(SAVE_DIR, name), frame)
    saved_by_sub[sub] = saved_by_sub.get(sub, 0) + 1
    print('  저장 %s   (%s 누적 %d장)' % (name, sub, saved_by_sub[sub]))
    cnt += 1
    last_saved = small(frame)
    msg, msg_t = 'SAVED  ' + name, time.time()


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    shot = frame.copy()          # 저장용 원본 (글씨도 반전도 들어가면 안 된다)

    # 화면만 거울처럼 뒤집는다. 저장되는 사진은 카메라가 본 그대로 남는다.
    # (학습에는 어느 쪽이든 상관없다 — fliplr=0.5 증강이 절반을 뒤집어 준다)
    if MIRROR:
        frame = cv2.flip(frame, 1)

    # ── 자동 촬영 판단 ──────────────────────────────────────
    now = time.time()
    if auto and now >= t_next:
        cur = small(frame)
        if last_saved is None or DIFF_TH <= 0:
            save(shot)
        else:
            diff = float(np.abs(cur - last_saved).mean())
            if diff >= DIFF_TH:
                save(shot)
            else:
                n_skip += 1
                # cv2.putText 는 한글을 못 그린다(??? 로 나온다). 화면 문구는 영어로 쓴다
                msg = 'SKIP (diff %.1f < %.1f) change your pose' % (diff, DIFF_TH)
                msg_t = now
        t_next = now + INTERVAL

    # ── 화면 표시 (저장본이 아니라 사본에 그린다) ────────────
    view = frame
    left = max(0.0, t_next - now)
    bar = 'AUTO %.1fs' % left if auto else 'PAUSED - press 1/2/3/4/0 to start'
    head = '[%s] %s  #%04d  %s' % (sub.upper(), bar, cnt, TAG)

    if GUIDE:
        # 가운데 세로선. 두 손을 이 선 좌우로 하나씩 두면 실제 게임과 같은 배치가 된다
        cv2.line(view, (WIDTH // 2, 26), (WIDTH // 2, HEIGHT - 26), (90, 90, 90), 1)

    cv2.rectangle(view, (0, 0), (WIDTH, 26), (0, 0, 0), -1)
    cv2.putText(view, head, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255) if auto else (0, 0, 255), 2)

    if msg and now - msg_t < 1.2:
        color = (0, 255, 0) if msg.startswith('SAVED') else (0, 165, 255)
        cv2.rectangle(view, (0, HEIGHT - 26), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.putText(view, msg, (6, HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    cv2.imshow('capture', view)

    # ── 키 처리 ─────────────────────────────────────────────
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        auto = not auto
        t_next = time.time() + INTERVAL
    elif key == ord('s'):
        save(shot)
        t_next = time.time() + INTERVAL
    elif key in SUBS:
        # 클래스 키를 누르면 대상을 바꾸고 바로 촬영을 시작한다
        sub = SUBS[key]
        last_saved = None      # 대상이 바뀌었으니 비슷함 검사를 새로 시작
        auto = True
        t_next = time.time() + INTERVAL
        print('  -> %s 촬영 시작 (%.1f초마다)' % (sub, INTERVAL))

cap.release()
cv2.destroyAllWindows()

print('-' * 62)
total = sum(saved_by_sub.values())
print(' 이번에 찍은 사진 : %d장' % total)
for k in sorted(saved_by_sub):
    print('   %-10s %d장' % (k, saved_by_sub[k]))
print(' 저장 위치 : %s' % os.path.abspath(SAVE_DIR))
if n_skip:
    rate = n_skip * 100.0 / (n_skip + total) if (n_skip + total) else 0
    print(' 너무 비슷해서 건너뜀 : %d번 (%.0f%%)' % (n_skip, rate))
    if rate > 50:
        print('   -> 절반 넘게 버렸다. 자세를 더 자주 바꾸거나 DIFF_TH를 낮춰라')
    elif rate < 5:
        print('   -> 거의 안 버렸다. 이대로 좋다')
print('-' * 62)
print(' 다음 할 일')
print('  1) bg로 찍은 사진은 라벨을 만들지 않는다 (그것이 background 역할)')
print('  2) 나머지는 LabelImg로 손마다 박스를 그린다')
print('  3) 기존 82장과 합쳐서 다시 학습한다')
print('-' * 62)
