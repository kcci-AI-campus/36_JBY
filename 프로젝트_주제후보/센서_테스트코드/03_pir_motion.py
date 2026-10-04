#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03. PIR 인체감지 센서 (HC-SR501) — 후보 H의 '트리거' 역할.

배선:
    VCC -> 5V  (물리 2번 핀)
    GND -> GND (물리 6번 핀)
    OUT -> GPIO22 (물리 15번 핀)

HC-SR501은 출력이 3.3V라 분압 없이 그대로 연결해도 된다.

센서 본체의 조절 나사 두 개:
    - Sx (감도)  : 감지 거리 3~7m 조절
    - Tx (지연)  : 감지 후 HIGH를 유지하는 시간 (수 초 ~ 5분)
    지연을 길게 두면 트리거가 계속 붙어 있어 측정이 왜곡된다.
    실험 전에 가장 짧게(반시계 끝) 돌려 둘 것.

전원을 넣고 나서 약 30~60초간은 센서가 스스로 보정을 하며
오감지가 잦다. 그 시간이 지난 뒤 측정한다.
"""
from gpiozero import MotionSensor
import sys, time

PIR_PIN = 22          # BCM 번호. 물리 15번 핀
WARMUP = 30           # 초. 센서 안정화 대기

def main():
    pir = MotionSensor(PIR_PIN)
    print("PIR 센서 안정화 대기 %d초... (이 동안 센서 앞에서 움직이지 마세요)" % WARMUP)
    time.sleep(WARMUP)
    print("준비 완료. 센서 앞에서 움직여 보세요. Ctrl+C 로 종료.\n")

    count = 0
    try:
        while True:
            pir.wait_for_motion()
            count += 1
            t0 = time.time()
            print("  [%d] %s  감지" % (count, time.strftime("%H:%M:%S")), end="", flush=True)
            pir.wait_for_no_motion()
            print("   -> 해제 (%.1f초 유지)" % (time.time() - t0))
    except KeyboardInterrupt:
        pass
    finally:
        pir.close()
        print("\n총 %d회 감지." % count)
        print("감지가 안 되면: ① 안정화 시간 부족 ② Tx 지연 나사가 길게 돌아가 있음")
        print("               ③ OUT 핀 배선 확인 ④ VCC가 5V인지 확인")

if __name__ == "__main__":
    sys.exit(main())
