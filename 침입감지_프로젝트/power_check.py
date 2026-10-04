# -*- coding: utf-8 -*-
"""
라즈베리파이 5 소비 전력 측정 — 추가 장비 없이

왜 필요한가
    detect_baby.py 가 6.5 W 를 쓴다는 것만으로는 아무 말도 못 한다.
    보드는 아무것도 안 해도 전력을 쓰기 때문이다.
    **아무것도 안 돌릴 때(대기)** 를 재야 "AI 가 얼마를 더 쓰는가" 를 말할 수 있다.

원리
    라즈베리파이 5 의 전원관리칩(PMIC)이 각 전원 레일의 전압과 전류를 알려준다.
        vcgencmd pmic_read_adc
    전압 x 전류 를 모두 더하면 보드가 쓰는 와트가 된다.

한계 (보고서에 반드시 같이 적을 것)
    메인 5V 레일의 '전류' 는 PMIC 가 잴 수 없다. 전압만 안다.
    그래서 절대값은 실제 소비보다 낮게 나온다.
    다만 빠지는 몫은 조건이 바뀌어도 거의 일정하므로
    **조건끼리의 차이** 를 비교하는 데는 유효하다. 우리가 필요한 것이 그 차이다.
    절대값이 필요하면 USB-C 전력계가 있어야 한다.

쓰는 법
    python3 power_check.py                      # 30초 재기
    python3 power_check.py --seconds 60
    python3 power_check.py --label 대기          # 이름을 붙여 기록
    python3 power_check.py --rails              # 레일별로 어디서 쓰는지 본다
    python3 power_check.py --compare            # 기록해 둔 것들을 표로 비교

측정 순서 (보고서용)
    1) 대기    : 감시 프로그램을 끄고
                 python3 power_check.py --seconds 60 --label 대기
    2) 게이팅끔 : python3 detect_baby.py <모델> --seconds 60 --no-gui
    3) 게이팅켬 : python3 detect_baby.py <모델> --seconds 60 --no-gui --gate
       (2,3 은 같은 장면이어야 한다)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

LOG = "power_log.json"


def read_rails():
    """PMIC 에서 레일별 (전압, 전류) 를 읽는다. 못 읽으면 None."""
    try:
        out = subprocess.run(["vcgencmd", "pmic_read_adc"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception as e:
        print("vcgencmd 를 실행하지 못했습니다 :", e)
        return None
    amps, volts = {}, {}
    for m in re.finditer(r"(\S+?)_([AV])\s+\w+\(\d+\)=([\d.]+)[AV]", out):
        name, kind, val = m.group(1), m.group(2), float(m.group(3))
        (amps if kind == "A" else volts)[name] = val
    if not amps:
        print("PMIC 출력을 해석하지 못했습니다. 라즈베리파이 5 가 맞습니까?")
        return None
    return {n: (volts[n], a) for n, a in amps.items() if n in volts}


def total_w(rails):
    return sum(v * a for v, a in rails.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=30.0, help="잴 시간(초)")
    p.add_argument("--label", default="", help="이 측정에 붙일 이름 (예: 대기)")
    p.add_argument("--rails", action="store_true", help="레일별 소비를 보여준다")
    p.add_argument("--compare", action="store_true", help="기록해 둔 측정을 표로 비교")
    a = p.parse_args()

    if a.compare:
        if not os.path.exists(LOG):
            print("아직 기록이 없습니다. --label 을 붙여 몇 번 재 보세요.")
            return
        rows = json.load(open(LOG, encoding="utf-8"))
        print()
        print("=" * 62)
        print(" 전력 비교 (PMIC 실측)")
        print("=" * 62)
        print("  %-16s %8s %8s %8s  %s" % ("이름", "평균W", "최소W", "최대W", "잰 시간"))
        print("  " + "-" * 58)
        base = None
        for r in rows:
            if base is None:
                base = r["avg"]
            print("  %-16s %8.2f %8.2f %8.2f  %.0f초"
                  % (r["label"], r["avg"], r["min"], r["max"], r["seconds"]))
        if len(rows) > 1:
            print()
            print("  첫 줄(%s)을 기준으로 한 차이" % rows[0]["label"])
            for r in rows[1:]:
                d = r["avg"] - base
                print("    %-16s %+.2f W  (%+.1f%%)"
                      % (r["label"], d, 100.0 * d / base if base else 0))
        print("=" * 62)
        return

    first = read_rails()
    if first is None:
        sys.exit(1)

    print()
    print("%.0f초 동안 1초마다 잽니다. 그동안 보드를 건드리지 마세요." % a.seconds)
    if a.label:
        print("이름 :", a.label)
    print()

    samples, rail_sum, t0 = [], {}, time.time()
    while time.time() - t0 < a.seconds:
        rails = read_rails()
        if rails:
            w = total_w(rails)
            samples.append(w)
            for n, (v, i) in rails.items():
                rail_sum[n] = rail_sum.get(n, 0.0) + v * i
            left = a.seconds - (time.time() - t0)
            sys.stdout.write("\r  %5.2f W   (남은 시간 %3.0f초, 표본 %d)"
                             % (w, max(0, left), len(samples)))
            sys.stdout.flush()
        time.sleep(1.0)
    print()

    if not samples:
        print("표본을 하나도 못 얻었습니다.")
        sys.exit(1)

    avg = sum(samples) / len(samples)
    print()
    print("=" * 52)
    print(" 전력 측정 결과  %s" % (a.label or ""))
    print("=" * 52)
    print("  평균        : %.2f W" % avg)
    print("  최소 / 최대 : %.2f W / %.2f W" % (min(samples), max(samples)))
    print("  표본        : %d 회 (%.0f초)" % (len(samples), a.seconds))

    if a.rails:
        print()
        print("  ── 레일별 평균 (어디서 쓰는가) ──")
        for n, tot in sorted(rail_sum.items(), key=lambda kv: -kv[1]):
            m = tot / len(samples)
            print("    %-14s %6.3f W   (%4.1f%%)" % (n, m, 100.0 * m / avg))
        print("    ※ 메인 5V 레일은 전류를 못 재서 여기에 안 나온다.")

    print("=" * 52)

    if a.label:
        rows = []
        if os.path.exists(LOG):
            try:
                rows = json.load(open(LOG, encoding="utf-8"))
            except Exception:
                rows = []
        rows = [r for r in rows if r["label"] != a.label]     # 같은 이름은 덮어쓴다
        rows.append({"label": a.label, "avg": avg,
                     "min": min(samples), "max": max(samples),
                     "seconds": a.seconds, "n": len(samples),
                     "when": time.strftime("%Y-%m-%d %H:%M")})
        json.dump(rows, open(LOG, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print()
        print("  %s 에 기록했습니다. 비교하려면 :  python3 power_check.py --compare" % LOG)


if __name__ == "__main__":
    main()
