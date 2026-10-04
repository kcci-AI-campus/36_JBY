#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""00. GPIO 환경 확인 — 배선하기 전에 먼저 돌린다.

라즈베리파이 5는 GPIO가 RP1 칩에 붙어 있어 기존 RPi.GPIO가 동작하지 않는다.
이 스크립트는 gpiozero(재단 권장)가 제대로 올라오는지, 어떤 백엔드를 쓰는지 확인한다.
"""
import sys


def main():
    print("=" * 56)
    print(" GPIO 환경 확인")
    print("=" * 56)
    print("python :", sys.version.split()[0])

    # 보드 모델 확인
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().decode("utf-8", "ignore").strip("\x00").strip()
        print("board  :", model)
        if "Raspberry Pi 5" in model:
            print("         -> Pi 5 확인. RPi.GPIO는 쓰지 말 것 (gpiozero 사용)")
    except OSError:
        print("board  : (확인 불가 - 라즈베리파이가 아닌 환경일 수 있음)")

    # gpiozero
    try:
        import gpiozero
        print("gpiozero:", gpiozero.__version__)
    except ImportError:
        print("gpiozero: 없음")
        print("  설치:  sudo apt install python3-gpiozero python3-lgpio")
        return 1

    # 실제로 핀 하나를 잡아 본다 (백엔드가 살아있는지 확인)
    from gpiozero import Device
    try:
        Device.ensure_pin_factory()
        print("backend :", type(Device.pin_factory).__name__)
    except Exception as e:
        print("backend : 실패 ->", e)
        print("  대부분 lgpio 미설치가 원인:  sudo apt install python3-lgpio")
        return 1

    # 핀 배치 출력
    try:
        from gpiozero import pi_info
        print()
        print(pi_info())
    except Exception:
        pass

    print()
    print("정상이면 다음 단계:  python3 01_led_blink.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
