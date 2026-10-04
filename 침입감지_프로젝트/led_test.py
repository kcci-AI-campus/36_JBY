# -*- coding: utf-8 -*-
"""
LED 배선이 맞는지 먼저 확인하는 시험 도구.

왜 따로 만드나
    추론 코드 안에서 LED 를 고치려 하면, 안 켜졌을 때
    배선이 틀린 건지 / 핀 번호가 틀린 건지 / 코드가 틀린 건지 구분이 안 된다.
    LED 만 따로 켜 보면 원인이 하나로 좁혀진다.

준비
    pip install gpiozero lgpio

실행
    python3 led_test.py              # 배선 안내 + 하나씩 순서대로 켜기
    python3 led_test.py --cycle      # 초록 -> 파랑 -> 빨강 상태를 반복 (시연 확인용)
    python3 led_test.py --rgb        # RGB LED 한 개짜리 배선일 때
    python3 led_test.py --pins 5,6,13   # 핀 번호를 바꿔서

핀 번호는 BCM 번호(GPIO17 의 17)이지 보드의 물리 핀 번호가 아니다. 아래 표 참고.
"""
import sys
import time
import argparse

# 윈도우 콘솔에서도 한글·표 문자가 깨지지 않게 (리눅스에서는 영향 없음)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# LED 3개를 따로 꽂는 경우. 가운데 자리는 파랑을 쓴다.
#   빨강/초록/파랑 3개로는 노랑을 만들 수 없다. 빨강+초록이 노랑이 되는 것은
#   세 색이 한 알에 들어 광학적으로 섞일 때뿐이고, 따로 떨어져 있으면
#   그냥 두 개가 켜질 뿐이다. 그래서 가운데 단계를 파랑으로 쓴다.
PINS = {"green": 17, "mid": 27, "red": 22}       # mid = 파랑 (노랑을 꽂았어도 그대로 동작)

# BCM 번호 -> 40핀 헤더의 물리 위치
PHYS = {17: 11, 27: 13, 22: 15, 5: 29, 6: 31, 13: 33, 16: 36, 19: 35, 26: 37,
        4: 7, 18: 12, 23: 16, 24: 18, 25: 22, 12: 32, 20: 38, 21: 40}


WIRING = """
==============================================================================
 배선 (LED 3개를 따로 쓰는 경우)
==============================================================================

 라즈베리파이 40핀 헤더 — 홀수 번호 줄만 쓴다.

 핀 위치가 헷갈리면 보드에서 이것부터 실행하면 그림으로 보여 준다 :
     pinout

 그 그림 기준으로
   * 1번 핀은 USB·랜 단자에서 **가장 먼 쪽 끝**이다 (USB-C 전원·SD카드 쪽)
   * 홀수 줄(1,3,5,...)은 **RAM 칩에 가까운 안쪽 줄**, 바깥 줄이 짝수다
   * 1번이 3.3V 이므로 테스터로도 확인할 수 있다

 1번부터 홀수 줄로 세어 **5·6·7·8번째**가 우리가 쓸 핀이다.

     1  3  5  7  [9] [11] [13] [15] 17 19 ...
                 GND  초록   파랑   빨강

        핀9    핀11     핀13     핀15
        GND   GPIO17   GPIO27   GPIO22
         │      │        │        │
         │      └─[220Ω]─┼────────┼──▶ 초록 LED 긴다리
         │               └────────┼──▶ 파랑 LED 긴다리   (각각 저항 하나씩)
         │                        └──▶ 빨강 LED 긴다리
         │
         └──────── 세 LED 의 짧은다리를 모두 여기로 (공통 GND)

 브레드보드에서는 이렇게 하면 편하다
   1) 브레드보드의 - 줄(파란 줄)에 핀9(GND)를 점퍼로 연결
   2) LED 3개를 꽂는다. **긴 다리가 +, 짧은 다리가 -**
   3) 각 LED 의 짧은 다리를 - 줄에 연결
   4) 각 LED 의 긴 다리에 저항(220~330Ω)을 직렬로 넣고, 그 끝을 핀11/13/15 에

 지켜야 할 것
   * 저항을 반드시 넣는다. 없으면 LED 와 GPIO 가 상한다
   * 라즈베리파이 GPIO 는 **3.3V 전용**이다. 5V(핀2, 핀4)에 절대 연결하지 않는다
   * 한 핀에 16mA, 전체 50mA 가 한계다. LED 3개는 약 18mA 라 안전하다
   * 저항은 긴 다리 쪽이든 짧은 다리 쪽이든 상관없다(직렬이므로)

==============================================================================
 배선 (RGB LED 한 개를 쓰는 경우, --rgb)
==============================================================================

 다리가 4개인 LED. 제일 긴 다리가 공통이다.
 대부분 '공통 음극'(common cathode) 이라 공통 다리를 GND 에 넣는다.

        핀9(GND) ──── 제일 긴 다리 (공통)
        핀11 ─[220Ω]─ R (빨강)
        핀13 ─[220Ω]─ G (초록)
        핀15 ─[220Ω]─ B (파랑)

   * 공통 다리를 GND 에 넣었는데 아무것도 안 켜지면 '공통 양극'이다.
     그때는 공통 다리를 3.3V(핀1)에 넣고 --rgb-anode 를 붙여 실행한다
   * 한 알짜리 RGB 일 때만 빨강+초록으로 노랑을 만든다 (이 코드가 알아서 한다)
     LED 3개를 따로 꽂았으면 가운데는 파랑을 쓴다

==============================================================================
"""


def main():
    ap = argparse.ArgumentParser(description="LED 배선 시험")
    ap.add_argument("--cycle", action="store_true", help="상태 3가지를 반복해서 보여준다")
    ap.add_argument("--rgb", action="store_true", help="RGB LED 한 개 배선")
    ap.add_argument("--rgb-anode", action="store_true", help="RGB 가 공통 양극일 때")
    ap.add_argument("--pins", default="", help="BCM 핀 번호 3개, 예: 17,27,22")
    ap.add_argument("--wiring", action="store_true", help="배선 안내만 보고 끝낸다")
    a = ap.parse_args()

    print(WIRING)
    if a.wiring:
        return 0

    pins = PINS.copy()
    if a.pins:
        try:
            v = [int(x) for x in a.pins.split(",")]
            assert len(v) == 3
            keys = ["red", "green", "blue"] if a.rgb else ["green", "mid", "red"]
            pins = dict(zip(keys, v))
        except Exception:
            print("--pins 는 숫자 3개여야 합니다. 예: --pins 17,27,22")
            return 1
    elif a.rgb:
        pins = {"red": 17, "green": 27, "blue": 22}

    try:
        from gpiozero import LED
    except ImportError:
        print("gpiozero 가 없습니다. 설치하세요 :")
        print("    pip install gpiozero lgpio")
        return 1

    try:
        dev = {k: LED(v, active_high=not a.rgb_anode) for k, v in pins.items()}
    except Exception as e:
        print("핀을 열 수 없습니다 :", e)
        print()
        print("  - 다른 프로그램이 같은 핀을 쓰고 있지 않은지 (detect_baby.py 종료했는지)")
        print("  - pip install lgpio 를 했는지")
        print("  - 사용자가 gpio 그룹에 있는지 :  groups")
        return 1

    print("사용할 핀 (BCM -> 물리 핀 번호)")
    for k, v in pins.items():
        print("   %-7s GPIO%-3d -> 헤더 %s번 핀"
              % (k, v, PHYS.get(v, "?")))
    print()

    def all_off():
        for d in dev.values():
            d.off()

    try:
        if a.rgb:
            # 색 = 어느 다리를 켜는가
            COLORS = {"초록 (대기)": ["green"],
                      "노랑 (아기 감지)": ["red", "green"],   # 한 알짜리 RGB 라 섞인다
                      "빨강 (위험)": ["red"],
                      "파랑 (확인용)": ["blue"],
                      "흰색 (전부)": ["red", "green", "blue"]}
        else:
            COLORS = {"초록 (대기)": ["green"],
                      "파랑 (아기 감지)": ["mid"],
                      "빨강 (위험)": ["red"]}

        if not a.cycle:
            print("=" * 60)
            print(" 하나씩 켭니다. 안 켜지는 게 있으면 그 배선만 다시 보세요.")
            print("=" * 60)
            for name, ks in COLORS.items():
                all_off()
                for k in ks:
                    dev[k].on()
                print("  지금 켜짐 : %-18s (%s)"
                      % (name, ", ".join("GPIO%d" % pins[k] for k in ks)))
                time.sleep(2.0)
            all_off()
            print()
            print("전부 켜졌으면 배선 끝. 상태 전환을 보려면 :")
            print("    python3 led_test.py --cycle")
            return 0

        print("=" * 60)
        print(" 실제 동작 순서를 반복합니다. Ctrl+C 로 종료.")
        print("=" * 60)
        while True:
            for name, ks in list(COLORS.items())[:3]:
                all_off()
                for k in ks:
                    dev[k].on()
                print("  %s" % name)
                time.sleep(1.5)
    except KeyboardInterrupt:
        pass
    finally:
        all_off()
        for d in dev.values():
            d.close()
        print("\n정리 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
