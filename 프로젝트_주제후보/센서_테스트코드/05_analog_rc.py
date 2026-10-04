#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""05. 아날로그 센서를 ADC 없이 읽기 — 저항 + 커패시터로 '충전 시간'을 잰다.

라즈베리파이 GPIO는 디지털 입력만 된다. '전기가 있다(1) / 없다(0)'만 구분하고,
전압이 1.7V인지 2.4V인지 같은 연속값은 읽지 못한다(ADC가 칩 안에 없다).
아두이노와 다른 점이 여기다.

그래서 시간으로 우회한다.

배선 (조도센서 CDS 기준):
    3.3V (물리 1번 핀) -- CDS -- [노드] -- GPIO25 (물리 22번 핀)
                                    |
                              1uF 커패시터
                                    |
                                   GND (물리 25번 핀)

원리:
    1) GPIO를 출력 LOW로 두어 커패시터를 완전히 비운다.
    2) GPIO를 입력으로 바꾸고, 전압이 HIGH로 인식될 때까지 시간을 센다.
    3) 어두우면 CDS 저항이 커져 천천히 차고, 밝으면 빨리 찬다.
       -> 충전 시간이 곧 밝기의 척도가 된다.

gpiozero의 LightSensor 클래스가 이 과정을 그대로 구현해 두었다.
값은 0.0(어두움) ~ 1.0(밝음)으로 정규화되어 나온다.

정밀도는 낮지만 '밝다/어둡다' 판정에는 충분하고, 가진 저항·커패시터만으로 된다.
정확한 값이 필요하면 MCP3008 같은 ADC 칩을 SPI로 붙이는 것이 정석이다.
"""
from gpiozero import LightSensor
import sys, time

PIN = 25              # BCM. 물리 22번
CHARGE_LIMIT = 0.1    # 초. 이 시간 안에 안 차면 '완전히 어두움'으로 본다
THRESHOLD = 0.5       # 이 값 이상이면 '밝음'

def main():
    ls = LightSensor(PIN, queue_len=5, charge_time_limit=CHARGE_LIMIT,
                     threshold=THRESHOLD)
    print("조도 측정 (0.0 어두움 ~ 1.0 밝음). 센서를 손으로 가려 보세요.")
    print("Ctrl+C 로 종료.\n")
    try:
        while True:
            v = ls.value
            bar = "#" * int(v * 40)
            state = "밝음" if v >= THRESHOLD else "어두움"
            print("\r %.3f |%-40s| %s" % (v, bar, state), end="", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        ls.close()
        print("\n\n값이 0 또는 1에서 안 움직이면:")
        print("  ① 커패시터 용량 조정 — 너무 작으면(0.1uF) 순식간에 차서 항상 1")
        print("     너무 크면(10uF) 늘 시간 초과라 항상 0. 1uF 권장")
        print("  ② CHARGE_LIMIT 값을 0.01 ~ 0.5 사이에서 조정")
        print("  ③ 배선 순서 확인 (3.3V - 센서 - 노드 - GPIO / 노드 - 커패시터 - GND)")

if __name__ == "__main__":
    sys.exit(main())
