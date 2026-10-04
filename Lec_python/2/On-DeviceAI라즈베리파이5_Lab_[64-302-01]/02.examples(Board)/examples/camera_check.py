# -*- coding: utf-8 -*-
"""
카메라가 실제로 초당 몇 장을 주는지 확인하는 진단 도구.

EX_13_opt.py 측정에서 "실제 새 프레임 초당 18.5장"이 나왔다.
웹캠은 보통 30장을 줄 수 있으므로, 왜 18.5인지 원인을 찾는다.

추론도 손 검출도 하지 않고 오직 cap.read()만 반복해서 잰다.

실행:  python3 camera_check.py
"""
import time
import cv2


def fourcc_to_str(v):
    v = int(v)
    return ''.join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def show_settings(cap, label):
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    print('  %-22s %dx%d  드라이버가 말하는 FPS=%.1f  형식=%s'
          % (label, w, h, fps, cc))


def measure(cap, seconds=5.0):
    """순수하게 읽기만 반복해서 초당 몇 장인지 잰다."""
    # 예열
    for _ in range(10):
        cap.read()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
    dt = time.perf_counter() - t0
    return n / dt


def trial(label, setup):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('  %-22s 카메라를 열 수 없습니다' % label)
        return None
    setup(cap)
    show_settings(cap, label)
    fps = measure(cap)
    print('  %-22s -> 실측 %.1f FPS' % ('', fps))
    cap.release()
    time.sleep(0.3)
    return fps


print('=' * 66)
print(' 카메라 진단 (각 조건마다 5초씩 읽기만 반복)')
print('=' * 66)

results = {}

results['현재 설정(320x240)'] = trial('현재 설정(320x240)', lambda c: (
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 320),
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240),
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)))

results['MJPG 형식'] = trial('MJPG 형식', lambda c: (
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 320),
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240),
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)))

results['FPS 30 요청'] = trial('FPS 30 요청', lambda c: (
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 320),
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240),
    c.set(cv2.CAP_PROP_FPS, 30),
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)))

results['MJPG + FPS 30'] = trial('MJPG + FPS 30', lambda c: (
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 320),
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240),
    c.set(cv2.CAP_PROP_FPS, 30),
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)))

results['자동노출 끔'] = trial('자동노출 끔', lambda c: (
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 320),
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240),
    c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1),   # 1=수동 (드라이버마다 다름)
    c.set(cv2.CAP_PROP_EXPOSURE, 50),
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)))

print('=' * 66)
print(' 요약')
print('-' * 66)
best_name, best_fps = None, 0
for k, v in results.items():
    if v is None:
        continue
    print('  %-22s %.1f FPS' % (k, v))
    if v > best_fps:
        best_name, best_fps = k, v
print('-' * 66)
if best_fps:
    print('  가장 빠른 조건: %s (%.1f FPS)' % (best_name, best_fps))
    if best_fps > 25:
        print('  -> 카메라는 더 낼 수 있다. EX_13_opt.py에 이 설정을 넣으면')
        print('     실제 처리량이 올라간다.')
    else:
        print('  -> 어떤 설정으로도 %.0f FPS 근처가 한계다.' % best_fps)
        print('     조명을 밝게 해 보고(자동노출이 어두우면 프레임률을 낮춘다)')
        print('     그래도 같으면 이 웹캠 자체의 상한이다.')
print('=' * 66)
