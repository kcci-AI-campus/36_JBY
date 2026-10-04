#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01. LED 점멸 — 출력이 되는지 확인한다.

배선:
    GPIO17 (물리 11번 핀) -> 330옴 저항 -> LED 긴 다리(+)
    LED 짧은 다리(-)      -> GND (물리 9번 핀)

저항을 빼먹으면 LED와 GPIO 핀이 상한다. 반드시 넣는다.
LED가 안 켜지면 다리 방향(긴 쪽이 +)부터 확인할 것.
"""
from gpiozero import LED
from signal import pause
import sys

LED_PIN = 17          # BCM 번호. 물리 11번 핀
INTERVAL = 0.5        # 초

def main():
    led = LED(LED_PIN)
    print("GPIO%d 에 연결된 LED를 %.1f초 간격으로 점멸합니다." % (LED_PIN, INTERVAL))
    print("Ctrl+C 로 종료.")
    led.blink(on_time=INTERVAL, off_time=INTERVAL)
    try:
        pause()
    except KeyboardInterrupt:
        pass
    finally:
        led.off()
        led.close()
        print("\n종료. LED가 깜빡였다면 출력 배선 정상 -> 02_button_in.py")

if __name__ == "__main__":
    sys.exit(main())
