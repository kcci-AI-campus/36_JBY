# -*- coding: utf-8 -*-
"""
배경 진단기 — "왜 흰 벽에서는 잘 되고 밤색 티 앞에서는 안 되나"를 숫자로 확인한다.

학습 데이터(RPS_Dataset_YOLO 104장)를 직접 분석해서 나온 값:

    배경 밝기 V (0~255)  :  최소 131,  평균 153,  최대 176   ← 104장 전부 이 사이
    배경 평균색 BGR      :  B 140  G 144  R 146             ← 거의 회색(무채색)
    손(박스 안) 평균색   :  B 131  G 139  R 160             ← R이 21 높다(살색)

즉 모델이 82장에서 배운 것은
    "밝고 무채색인 벽 앞에 놓인, 붉은기가 도는 덩어리 = 손"
이다. 밤색 티는 (1) 어둡고 (2) 붉다 → 두 단서가 동시에 무너진다.

이 스크립트는 지금 카메라에 보이는 배경이 그 범위 안에 있는지 알려 준다.
화면 창을 띄우지 않으므로 SSH만으로도 돌아간다.

실행:  python3 bg_check.py          (Ctrl+C 로 종료)
"""
import time
import cv2
import numpy as np

# ── 학습 데이터에서 실측한 값 (이 숫자가 판단 기준) ──────────
TRAIN_V_MIN, TRAIN_V_MAX = 131, 176
TRAIN_V_MEAN = 153
TRAIN_BG_BGR = (140, 144, 146)

FOURCC = 'MJPG'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print('=' * 64)
print(' 배경 진단  —  학습 배경은 V %d~%d (평균 %d), 거의 무채색'
      % (TRAIN_V_MIN, TRAIN_V_MAX, TRAIN_V_MEAN))
print(' 손은 화면에서 빼고, 배경만 비춘 상태로 재세요.')
print(' Ctrl+C 로 종료')
print('=' * 64)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = float(hsv[:, :, 2].mean())
        b, g, r = [float(x) for x in frame.reshape(-1, 3).mean(0)]

        # 밝기 판정
        if v < TRAIN_V_MIN:
            v_msg = '너무 어둡다 (학습 최소 %d보다 %.0f 낮음)' % (
                TRAIN_V_MIN, TRAIN_V_MIN - v)
        elif v > TRAIN_V_MAX:
            v_msg = '너무 밝다 (학습 최대 %d보다 %.0f 높음)' % (
                TRAIN_V_MAX, v - TRAIN_V_MAX)
        else:
            v_msg = '학습 범위 안 (좋음)'

        # 색기울기 판정 : R이 B보다 많이 크면 '붉은 배경' = 살색과 헷갈린다
        tilt = r - b
        if tilt > 25:
            c_msg = '붉은 배경 (R-B=%.0f). 살색과 구분이 어렵다' % tilt
        elif tilt < -25:
            c_msg = '푸른 배경 (R-B=%.0f)' % tilt
        else:
            c_msg = '무채색에 가까움 (R-B=%.0f, 좋음)' % tilt

        print('V %5.1f  |  BGR %3.0f/%3.0f/%3.0f  |  %s  |  %s'
              % (v, b, g, r, v_msg, c_msg))
        time.sleep(1.0)

except KeyboardInterrupt:
    print()
    print('-' * 64)
    print(' 정리: 배경이 학습 범위(V %d~%d, 무채색)에서 멀수록 인식률이 떨어진다.'
          % (TRAIN_V_MIN, TRAIN_V_MAX))
    print(' 당장 할 수 있는 것 : 밝은 무채색 벽/종이 앞에서 하기, 조명 켜기')
    print(' 제대로 고치는 것   : 그 배경에서 사진을 찍어 학습 데이터에 추가하기')
    print('-' * 64)
finally:
    cap.release()
