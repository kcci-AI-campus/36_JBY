# -*- coding: utf-8 -*-
"""
트레이싱 맛보기 데모 (PC에서 카메라·보드 없이 바로 실행)

보드에서 쓸 trace_util.py가 제대로 도는지 확인하고,
Perfetto 뷰어가 어떤 모양인지 미리 보기 위한 것이다.

가짜 파이프라인을 10초쯤 돌려 trace_demo.json 을 만든다.
실행:  trace_demo.bat  (또는 .venv\\Scripts\\python.exe trace_demo.py)
"""
import os
import sys
import time

# 보드용 trace_util.py를 그대로 가져다 쓴다 (복사본을 만들지 않기 위해)
HERE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(
    HERE, '..', '2', 'On-DeviceAI라즈베리파이5_Lab_[64-302-01]',
    '02.examples(Board)', 'examples')
sys.path.append(os.path.abspath(TRACE_DIR))

try:
    from trace_util import Tracer
except ImportError:
    print('trace_util.py 를 찾지 못했습니다. 찾은 경로:')
    print('   ', os.path.abspath(TRACE_DIR))
    sys.exit(1)


def burn(ms):
    """ms 밀리초 동안 실제로 CPU를 쓴다 (sleep과 달리 코어 그래프에 나타난다)."""
    end = time.perf_counter() + ms / 1000.0
    x = 0
    while time.perf_counter() < end:
        x += 1
    return x


def main():
    out = os.path.join(HERE, 'trace_demo.json')
    tr = Tracer(out, sample_cpu=True, sample_hz=20)

    print('가짜 파이프라인을 200프레임 돌립니다 (약 10초)...')
    for i in range(200):
        with tr.span('frame'):
            with tr.span('camera'):
                time.sleep(0.005)          # USB 대기 흉내 (계산 아님)
            with tr.span('handDetect'):
                burn(22)                   # MediaPipe 흉내 (CPU 사용)
            with tr.span('inference'):
                burn(10)                   # 모델 추론 흉내 (CPU 사용)
            with tr.span('imshow'):
                time.sleep(0.006)          # 화면 전송 흉내
            with tr.span('waitKey'):
                time.sleep(0.010)          # 그냥 기다리는 구간
        if i % 40 == 0:
            tr.instant('no-hand')          # 손을 못 찾은 프레임 표시

    tr.save()


if __name__ == '__main__':
    main()
