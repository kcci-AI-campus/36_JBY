#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""02. 버튼 입력 — 입력이 되는지 확인한다.

배선:
    GPIO27 (물리 13번 핀) -- 버튼 -- GND (물리 14번 핀)

저항이 필요 없는 이유: gpiozero의 Button은 기본으로 라즈베리파이 내부의
'풀업 저항'을 켠다. 평소에는 핀이 3.3V로 끌어올려져 있다가(=1),
버튼을 누르면 GND와 이어져 0V가 된다(=0). 그래서 '눌림 = 0'이다.
"""
from gpiozero import Button
import sys, time

BTN_PIN = 27          # BCM 번호. 물리 13번 핀

def main():
    btn = Button(BTN_PIN, pull_up=True, bounce_time=0.05)
    count = 0
    print("GPIO%d 버튼을 눌러 보세요. Ctrl+C 로 종료." % BTN_PIN)
    try:
        while True:
            btn.wait_for_press()
            count += 1
            t0 = time.time()
            print("  [%d] 눌림" % count, end="", flush=True)
            btn.wait_for_release()
            print("  (%.2f초 유지)" % (time.time() - t0))
    except KeyboardInterrupt:
        pass
    finally:
        btn.close()
        print("\n총 %d회 감지. 1회 이상이면 입력 배선 정상 -> 03_pir_motion.py" % count)

if __name__ == "__main__":
    sys.exit(main())
