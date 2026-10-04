#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""04. 초음파 거리 센서 (HC-SR04) — 거리로 트리거를 걸 때 쓴다.

배선:
    VCC  -> 5V  (물리 4번 핀)
    GND  -> GND (물리 20번 핀)
    TRIG -> GPIO23 (물리 16번 핀)
    ECHO -> [ 1kR ] -+- GPIO24 (물리 18번 핀)
                     |
                  [ 2kR ]
                     |
                    GND

*** ECHO는 반드시 분압할 것 ***
ECHO 핀은 5V로 신호를 내보낸다. GPIO는 3.3V 전용이라 그대로 꽂으면 핀이 망가진다.
5V x 2k/(1k+2k) = 3.3V. 10k + 20k 조합도 비율이 같아 괜찮다.

동작 원리:
    TRIG에 10us 펄스를 주면 센서가 초음파를 쏘고, 반사음이 돌아올 때까지
    ECHO를 HIGH로 유지한다. 그 시간 x 음속(약 343m/s) / 2 = 거리.
    gpiozero의 DistanceSensor가 이 계산을 대신 해 준다(단위: 미터).
"""
from gpiozero import DistanceSensor
import sys, time

TRIG_PIN = 23         # BCM. 물리 16번
ECHO_PIN = 24         # BCM. 물리 18번
MAX_DIST = 2.0        # m. 이 거리 밖은 측정하지 않는다
TRIGGER_CM = 30.0     # 이 거리 안으로 들어오면 '접근'으로 본다

def main():
    sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN,
                            max_distance=MAX_DIST, queue_len=5)
    print("초음파 거리 측정. %.0fcm 안으로 들어오면 접근으로 표시합니다." % TRIGGER_CM)
    print("Ctrl+C 로 종료.\n")

    near = False
    events = 0
    try:
        while True:
            cm = sensor.distance * 100.0
            bar = "#" * min(40, int(cm / 5))
            mark = ""
            if cm < TRIGGER_CM and not near:
                near, events = True, events + 1
                mark = "   <== 접근 감지 (%d)" % events
            elif cm >= TRIGGER_CM + 5 and near:   # 5cm 여유(채터링 방지)
                near = False
            print("\r %6.1f cm |%-40s|%s" % (cm, bar, mark), end="", flush=True)
            if mark:
                print()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()
        print("\n\n접근 이벤트 %d회." % events)
        print("값이 계속 0 또는 최대치면: ① ECHO 분압 확인 ② TRIG/ECHO 바뀌지 않았는지 ③ VCC 5V 확인")

if __name__ == "__main__":
    sys.exit(main())
