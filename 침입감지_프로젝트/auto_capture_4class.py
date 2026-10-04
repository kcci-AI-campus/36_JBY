# -*- coding: utf-8 -*-
"""
자동 촬영기 (4클래스판) — 가위바위보용 EX_01_Auto_Capture.py를 이번 프로젝트에 맞게 고친 것.

프로젝트 : 저전력 온디바이스 AI 기반 위험구역 침입 감지 시스템
학습 클래스 : 0 baby(아기 인형)  1 adult(성인)  2 knife(칼)  3 outlet(콘센트)

┌───────────────────────────────────────────────────────────────┐
│ ⚠ 숫자 키로 고르는 것은 '라벨'이 아니라 '지금 어떤 상황을 찍는  │
│    중인지'를 파일 이름에 남기는 태그다.                        │
│    가위바위보 때는 한 장에 손 하나라 태그 = 클래스였지만,      │
│    이번에는 한 장에 아기 인형과 칼이 같이 나온다.              │
│    실제 라벨은 LabelImg에서 물체마다 박스를 그려 정한다.       │
└───────────────────────────────────────────────────────────────┘

파일 이름 = <조건>_<상황>_<번호>.jpg
    예)  front-sit-mid-bright_near_0042.jpg
    조건이 이름에 들어가므로, 나중에 **조건별 검출률**을 집계할 수 있다.
    ("측면·멀리·어두움에서 mAP가 얼마나 떨어지는가" → 보고서 분석 한 편)

조작
  [상황]  1 baby    2 adult   3 knife   4 outlet
          5 near    아기 + 위험물 가까이   ← 경보가 울려야 하는 상황
          6 edge    아기 + 위험물 애매한 거리 ← 판정이 갈리는 상황
          7 mix     여러 개 섞임
          0 bg      빈 장면 (라벨 만들지 않음. background 역할)

  [조건]  a  각도  front → side → deg45 → top
          p  자세  lie → sit → crawl → stand
          d  거리  near(~1m) → mid(1~2m) → far(2~3m)
          l  조명  bright → dim

  SPACE   자동 촬영 멈춤 / 재개
  s       지금 즉시 한 장
  q       끝내기

실행
    python3 auto_capture_4class.py
"""
import os
import time
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────────────────────
SAVE_DIR   = './captures'   # 저장 폴더. 없으면 만든다
INTERVAL   = 0.8            # 몇 초마다 한 장

# 직전 저장본과 얼마나 달라야 저장할지 (평균 밝기 차이, 0~255)
#   0    : 검사 안 함. 무조건 다 찍는다
#   3.0  : 카메라가 멈춰 있을 때 생기는 '진짜 중복'만 버린다  ← 권장
#   8.0  : 배치가 제법 바뀌어야 저장한다 (장수가 확 줄어든다)
#
# 왜 0으로 두지 않는가
#   ① 라벨링 비용 : 한 장마다 물체 여러 개에 박스를 그려야 한다.
#      거의 같은 사진이 쌓이면 그리는 시간만 늘고 배우는 것은 없다.
#   ② 시험지 오염 : 거의 같은 사진 두 장이 train과 test에 나뉘어 들어가면
#      이미 본 것을 시험 보는 셈이라 mAP가 실제보다 높게 나온다.
DIFF_TH    = 3.0
WIDTH      = 640
HEIGHT     = 480
FOURCC     = 'MJPG'
MIRROR     = False          # 인형을 손으로 옮기며 찍으므로 화면 반전은 끈다

# ── 상황 태그 (숫자 키) ──────────────────────────────────────
SUBS = {ord('1'): 'baby',     # 아기 인형 단독 (위험물에서 멀리)
        ord('2'): 'adult',    # 성인 단독
        ord('3'): 'knife',    # 칼만
        ord('4'): 'outlet',   # 콘센트·멀티탭만
        ord('5'): 'near',     # 아기 + 위험물 가까이   ← 경보 상황
        ord('6'): 'edge',     # 아기 + 위험물 애매한 거리 ← 판정 경계
        ord('7'): 'mix',      # 여러 개 섞임 (아기+성인+위험물 등)
        ord('0'): 'bg'}       # 빈 장면

# 상황별 목표 장수.
#   near / edge 를 가장 많이 찍는 이유 : 실제로 판정이 갈리는 상황이라
#   여기 데이터가 모자라면 경계에서 자꾸 틀린다.
TARGETS = {'baby':   80,
           'adult':  50,
           'knife':  30,
           'outlet': 30,
           'near':   90,
           'edge':   70,
           'mix':    40,
           'bg':     30}

# ── 조건 축 (알파벳 키로 순환) ───────────────────────────────
#   팀 회의에서 정한 촬영 조건을 그대로 옮긴 것.
CONDS = {
    'a': ('angle', ['front', 'side', 'deg45', 'top']),   # 천장 근처 설치 대비 top 포함
    'p': ('pose',  ['lie', 'sit', 'crawl', 'stand']),    # 인형을 세울 수 있으면 stand
    'd': ('dist',  ['near', 'mid', 'far']),              # ~1m / 1~2m / 2~3m
    'l': ('light', ['bright', 'dim']),                   # 형광등 켜고 끄는 정도면 충분
}
cond_idx = {k: 0 for k in CONDS}


def cond_tag():
    """현재 조건을 파일 이름용 문자열로. 예: front-sit-mid-bright"""
    return '-'.join(CONDS[k][1][cond_idx[k]] for k in ('a', 'p', 'd', 'l'))


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
        tail = os.path.splitext(f)[0].split('_')[-1]
        if tail.isdigit():
            n = max(n, int(tail))
    return n + 1


def count_existing():
    """폴더에 이미 있는 사진을 상황별로 센다. 이어서 찍을 때 진행률이 맞도록."""
    got = {}
    for f in os.listdir(SAVE_DIR):
        if not f.lower().endswith('.jpg'):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) >= 3 and parts[-2] in TARGETS:
            got[parts[-2]] = got.get(parts[-2], 0) + 1
    return got


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
n_skip = 0
sub = 'baby'
auto = False            # 멈춘 상태로 시작. 배치를 잡고 숫자 키를 눌러야 시작된다
t_next = time.time() + INTERVAL
last_saved = None
saved_by_sub = {}
have = count_existing()
msg, msg_t = '', 0.0

print('=' * 70)
print(' 저장 위치 : %s' % os.path.abspath(SAVE_DIR))
print(' 시작 번호 : %04d   (기존 파일 뒤에서 이어 붙인다)' % cnt)
print('-' * 70)
print(' [상황] 1 baby  2 adult  3 knife  4 outlet')
print('        5 near(아기+위험물 가까이)  6 edge(애매한 거리)  7 mix  0 bg')
print(' [조건] a 각도   p 자세   d 거리   l 조명   (누를 때마다 다음 값으로)')
print(' SPACE 멈춤/재개    s 수동 한 장    q 종료')
print('-' * 70)
if have:
    print(' 기존 보유량 :', ', '.join('%s %d' % (k, v) for k, v in sorted(have.items())))
print(' 현재 조건 : %s' % cond_tag())
print(' >> 멈춘 상태로 시작합니다. 배치를 잡고 창을 클릭한 뒤 숫자 키를 누르세요. <<')
print('=' * 70)


def small(img):
    """비슷한지 비교할 때 쓰는 축소판. 작게 줄이면 잡음에 덜 흔들린다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 48)).astype(np.float32)


def total_of(s):
    """이 상황으로 지금까지 확보한 총 장수 (기존 + 이번)."""
    return have.get(s, 0) + saved_by_sub.get(s, 0)


def save(frame):
    global cnt, last_saved, msg, msg_t
    name = '%s_%s_%04d.jpg' % (cond_tag(), sub, cnt)
    cv2.imwrite(os.path.join(SAVE_DIR, name), frame)
    saved_by_sub[sub] = saved_by_sub.get(sub, 0) + 1
    tgt = TARGETS.get(sub, 0)
    print('  저장 %s   (%s %d/%d)' % (name, sub, total_of(sub), tgt))
    if tgt and total_of(sub) == tgt:
        print('  >>> %s 목표 %d장 달성. 다음 상황으로 넘어가세요. <<<' % (sub, tgt))
    cnt += 1
    last_saved = small(frame)
    msg, msg_t = 'SAVED  ' + name, time.time()


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    shot = frame.copy()          # 저장용 원본 (글씨도 반전도 들어가면 안 된다)

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
                msg = 'SKIP (diff %.1f < %.1f) move the objects' % (diff, DIFF_TH)
                msg_t = now
        t_next = now + INTERVAL

    # ── 화면 표시 (저장본이 아니라 사본에 그린다) ────────────
    view = frame
    left = max(0.0, t_next - now)
    bar = 'AUTO %.1fs' % left if auto else 'PAUSED - press 1~7/0'
    tgt = TARGETS.get(sub, 0)
    head = '[%s %d/%d] %s  #%04d' % (sub.upper(), total_of(sub), tgt, bar, cnt)

    cv2.rectangle(view, (0, 0), (WIDTH, 46), (0, 0, 0), -1)
    cv2.putText(view, head, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255) if auto else (0, 0, 255), 2)
    cv2.putText(view, cond_tag() + '   (a/p/d/l to change)', (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 진행 막대 — 이 상황을 얼마나 채웠는지 한눈에
    if tgt:
        ratio = min(1.0, total_of(sub) / float(tgt))
        cv2.rectangle(view, (0, 46), (WIDTH, 50), (40, 40, 40), -1)
        cv2.rectangle(view, (0, 46), (int(WIDTH * ratio), 50), (0, 200, 0), -1)

    if msg and now - msg_t < 1.2:
        color = (0, 255, 0) if msg.startswith('SAVED') else (0, 165, 255)
        cv2.rectangle(view, (0, HEIGHT - 26), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.putText(view, msg, (6, HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

    cv2.imshow('capture', view)

    # ── 키 처리 ─────────────────────────────────────────────
    key = cv2.waitKey(10) & 0xFF
    ch = chr(key) if 32 <= key < 127 else ''      # 조건 키(a/p/d/l) 판별용
    if key == ord('q'):
        break
    elif key == ord(' '):
        auto = not auto
        t_next = time.time() + INTERVAL
    elif key == ord('s'):
        save(shot)
        t_next = time.time() + INTERVAL
    elif ch in CONDS:
        k = ch
        name, values = CONDS[k]
        cond_idx[k] = (cond_idx[k] + 1) % len(values)
        last_saved = None      # 조건이 바뀌었으니 비슷함 검사를 새로 시작
        print('  [조건] %-5s -> %s   (전체: %s)' % (name, values[cond_idx[k]], cond_tag()))
    elif key in SUBS:
        sub = SUBS[key]
        last_saved = None      # 상황이 바뀌었으니 비슷함 검사를 새로 시작
        auto = True
        t_next = time.time() + INTERVAL
        print('  -> %s 촬영 시작 (%.1f초마다, 목표 %d장, 조건 %s)'
              % (sub, INTERVAL, TARGETS.get(sub, 0), cond_tag()))

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────
#  마무리 요약
# ─────────────────────────────────────────────────────────────
print('-' * 70)
total = sum(saved_by_sub.values())
print(' 이번에 찍은 사진 : %d장' % total)
print('')
print(' %-8s %8s %8s %8s' % ('상황', '이번', '누적', '목표'))
for k in sorted(TARGETS):
    print(' %-8s %8d %8d %8d%s'
          % (k, saved_by_sub.get(k, 0), total_of(k), TARGETS[k],
             '   OK' if total_of(k) >= TARGETS[k] else ''))
print('')
print(' 저장 위치 : %s' % os.path.abspath(SAVE_DIR))
if n_skip:
    rate = n_skip * 100.0 / (n_skip + total) if (n_skip + total) else 0
    print(' 너무 비슷해서 건너뜀 : %d번 (%.0f%%)' % (n_skip, rate))
    if rate > 50:
        print('   -> 절반 넘게 버렸다. 물체를 더 자주 옮기거나 DIFF_TH를 낮춰라')
    elif rate < 5:
        print('   -> 거의 안 버렸다. 이대로 좋다')
print('-' * 70)
print(' 다음 할 일')
print('  1) bg 사진은 라벨 파일을 만들지 않는다 (그것이 background 역할)')
print('  2) 자동 라벨을 먼저 시도한다 — COCO에 person, knife, teddy bear 가 있다:')
print('       yolo predict model=yolo11n.pt source=captures/ save_txt=True conf=0.2')
print('     person -> 1(adult), knife -> 2(knife), teddy bear -> 0(baby) 로 번호만 바꾼다')
print('     (콘센트는 COCO에 없으므로 손으로 그린다)')
print('  3) 나머지는 LabelImg로 물체마다 박스를 그린다')
print('       클래스 순서 : 0 baby   1 adult   2 knife   3 outlet')
print('  4) 시험셋은 다른 날·다른 시간에 찍은 것으로 따로 뗀다')
print('-' * 70)
