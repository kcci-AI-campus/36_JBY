# -*- coding: utf-8 -*-
"""
카메라가 왜 초당 20장밖에 안 주는지 원인을 가리는 진단 도구.

bench_models.py 측정에서 세 모델 모두 정확히 20.0 FPS 로 막혔다.
추론 능력은 115.9 FPS(INT8)인데 카메라가 20장만 주기 때문이다.
원인 후보가 넷인데, 이 도구로 하나씩 지운다.

    ① cap.set() 이 무시됨      -> 요청값과 되읽은 값을 나란히 출력해 확인
    ② 자동 노출이 프레임률을 떨굼 -> 노출 고정 조건과 비교, 프레임 간격의 흔들림으로 판별
    ③ 이 웹캠의 실제 상한       -> v4l2-ctl 이 신고하는 지원 표가 결정적 증거
    ④ USB 대역폭               -> 해상도·형식을 낮췄을 때 올라가는지로 판별

추론도 그리기도 하지 않고 오직 cap.read() 만 반복해서 잰다.

실행
    source ~/work/env/bin/activate
    cd ~/work/baby
    python3 camera_check.py

    python3 camera_check.py --seconds 8     조건당 측정 시간(기본 5초)
    python3 camera_check.py --device 1      다른 카메라 번호

읽는 법은 맨 아래 출력에 함께 나온다.
"""
import sys
import time
import shutil
import argparse
import subprocess

import cv2

# 윈도우 콘솔에서도 한글·표 문자가 깨지지 않게 (리눅스에서는 영향 없음)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# ─────────────────────────────────────────────────────────────
#  카메라가 스스로 신고하는 지원 표 — 이게 가장 확실한 증거다
# ─────────────────────────────────────────────────────────────
def show_v4l2(dev):
    """v4l2-ctl 로 이 카메라가 어떤 형식·해상도에서 몇 FPS까지 되는지 본다."""
    print("=" * 72)
    print(" [0] 카메라가 신고하는 지원 목록  (v4l2-ctl)")
    print("=" * 72)
    if shutil.which("v4l2-ctl") is None:
        print("  v4l2-ctl 이 없습니다. 설치하면 결정적인 근거가 나옵니다 :")
        print("      sudo apt install -y v4l-utils")
        print()
        return
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", "/dev/video%d" % dev, "--list-formats-ext"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        print("  실행 실패 :", e)
        print()
        return

    # 640x480 줄과 그 아래 프레임률만 눈에 띄게 뽑아 준다
    print(out.strip() if out.strip() else "  (출력 없음)")
    print()
    print("  ↑ 여기서 MJPG 의 640x480 칸에 30.000 fps 가 적혀 있는데 실측이 20 이라면")
    print("    카메라 상한이 아니라 다른 이유(설정 무시 또는 노출)다.")
    print("    20.000 fps 로 적혀 있다면 이 카메라의 상한이 맞다.")
    print()


# ─────────────────────────────────────────────────────────────
def fourcc_to_str(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def readback(cap):
    """드라이버에 실제로 적용된 값을 되읽는다. 요청과 다르면 set 이 무시된 것."""
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
            fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)))


def measure(cap, seconds, want_bright=False):
    """읽기만 반복. 평균 FPS, 간격 흔들림, (원하면) 화면 밝기를 돌려준다."""
    for _ in range(10):            # 예열
        cap.read()
    gaps = []
    bright = []
    t_prev = time.perf_counter()
    t0 = t_prev
    while time.perf_counter() - t0 < seconds:
        ok, fr = cap.read()
        if not ok:
            break
        now = time.perf_counter()
        gaps.append((now - t_prev) * 1000.0)
        t_prev = now
        if want_bright and fr is not None and len(gaps) % 10 == 0:
            bright.append(float(fr.mean()))
    if not gaps:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    n = len(gaps)
    avg = sum(gaps) / n
    lo, hi = min(gaps), max(gaps)
    b = (sum(bright) / len(bright)) if bright else 0.0
    return n / (t_prev - t0), avg, lo, hi, b


def trial(label, dev, setup, seconds):
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print("  %-26s 카메라를 열 수 없습니다" % label)
        return None
    setup(cap)
    w, h, drv_fps, cc = readback(cap)
    fps, avg, lo, hi, _b = measure(cap, seconds)
    cap.release()
    time.sleep(0.3)

    print("  %-26s %dx%d %s  드라이버 신고 %.0f fps" % (label, w, h, cc, drv_fps))
    print("  %-26s 실측 %5.1f FPS   간격 평균 %.1f ms (최소 %.1f / 최대 %.1f)"
          % ("", fps, avg, lo, hi))
    return {"label": label, "w": w, "h": h, "fourcc": cc,
            "drv": drv_fps, "fps": fps, "avg": avg, "lo": lo, "hi": hi}


# ─────────────────────────────────────────────────────────────
def threaded_trial(label, dev, seconds):
    """읽기만 하는 스레드를 따로 두고, 본 루프는 최신 프레임을 가져다 쓴다.

    버퍼가 한 장뿐이면 본 루프가 조금만 늦어도 그 프레임이 버려지고
    다음 것을 기다리게 된다(그래서 30fps 짜리가 20fps 로 보인다).
    읽기를 전담하는 스레드가 계속 비워 주면 그런 누락이 없어진다.
    여기서는 **스레드가 실제로 몇 장을 받는지**를 센다.
    """
    import threading
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print("  %-26s 카메라를 열 수 없습니다" % label)
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w, h, drv_fps, cc = readback(cap)

    stop = threading.Event()
    gaps = []

    def loop():
        for _ in range(10):
            cap.read()
        t_prev = time.perf_counter()
        while not stop.is_set():
            ok, _ = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            gaps.append((now - t_prev) * 1000.0)
            t_prev = now

    th = threading.Thread(target=loop, daemon=True)
    t0 = time.perf_counter()
    th.start()
    time.sleep(seconds + 0.5)       # 예열 시간 포함
    stop.set()
    th.join(timeout=2.0)
    dt = time.perf_counter() - t0
    cap.release()
    time.sleep(0.3)

    if not gaps:
        print("  %-26s 프레임을 못 받았습니다" % label)
        return None
    n = len(gaps)
    avg, lo, hi = sum(gaps) / n, min(gaps), max(gaps)
    fps = 1000.0 / avg
    print("  %-26s %dx%d %s  드라이버 신고 %.0f fps" % (label, w, h, cc, drv_fps))
    print("  %-26s 실측 %5.1f FPS   간격 평균 %.1f ms (최소 %.1f / 최대 %.1f)"
          % ("", fps, avg, lo, hi))
    return {"label": label, "w": w, "h": h, "fourcc": cc,
            "drv": drv_fps, "fps": fps, "avg": avg, "lo": lo, "hi": hi}


def expo_sweep(dev, seconds):
    """노출 시간을 바꿔 가며 FPS 와 화면 밝기를 함께 잰다.

    노출을 짧게 하면 프레임은 빨라지지만 화면이 어두워진다.
    어두우면 검출률이 떨어지므로 **FPS 와 밝기의 맞바꿈**을 눈으로 봐야
    어디를 고를지 정할 수 있다. 숫자 하나만 보고 정하면 안 된다.

    값의 단위는 드라이버마다 다르다(보통 100마이크로초). 절대값보다
    '어느 값에서 FPS 가 꺾이는가' 를 보면 된다.
    """
    MJPG = cv2.VideoWriter_fourcc(*"MJPG")
    print("=" * 72)
    print(" [4] 노출값 훑기 — FPS 와 밝기의 맞바꿈")
    print("=" * 72)
    print("  %-14s %8s %10s %14s %s" % ("노출값", "FPS", "평균간격", "흔들림", "화면밝기(0~255)"))
    print("  " + "-" * 62)

    rows = []
    for expo in [0, 5, 10, 20, 39, 78, 156, 312, 625]:
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            print("  카메라를 열 수 없습니다")
            return []
        cap.set(cv2.CAP_PROP_FOURCC, MJPG)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if expo == 0:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # 3 = 자동 (V4L2)
            label = "자동"
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # 1 = 수동 (V4L2)
            cap.set(cv2.CAP_PROP_EXPOSURE, expo)
            label = str(expo)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps, avg, lo, hi, b = measure(cap, seconds, want_bright=True)
        cap.release()
        time.sleep(0.3)
        print("  %-14s %8.1f %8.1f ms %11s %8.1f%s"
              % (label, fps, avg, "%.0f~%.0f ms" % (lo, hi), b,
                 "   <- 너무 어둡다" if b < 40 else ""))
        rows.append({"expo": expo, "label": label, "fps": fps, "bright": b,
                     "avg": avg, "lo": lo, "hi": hi})

    # ── 판정 ──
    print()
    ok = [r for r in rows if r["bright"] >= 40]        # 쓸 만한 밝기만
    if ok:
        best = max(ok, key=lambda r: r["fps"])
        auto = next((r for r in rows if r["expo"] == 0), None)
        print("  화면이 쓸 만한(밝기 40 이상) 것 중 가장 빠른 조건 : 노출 %s -> %.1f FPS"
              % (best["label"], best["fps"]))
        if auto and best["fps"] > auto["fps"] * 1.15:
            print("  자동(%.1f FPS) 보다 %.0f%% 빠르다. detect_baby.py 에 이 값을 넣을 만하다."
                  % (auto["fps"], (best["fps"] / auto["fps"] - 1) * 100))
            print("      cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)")
            print("      cap.set(cv2.CAP_PROP_EXPOSURE, %s)" % best["label"])
        else:
            print("  자동과 큰 차이가 없다. 노출을 고정해도 이득이 없다.")
    else:
        print("  어느 노출값에서도 화면이 너무 어둡다. 조명을 켜고 다시 재라.")

    print()
    print("  ※ 노출을 짧게 하면 빨라지지만 어두워지고, 어두우면 검출률이 떨어진다.")
    print("     FPS 만 보고 고르지 말 것. 밝기 40 아래면 쓸 수 없다고 보면 된다.")
    print("=" * 72)
    return rows


def main():
    ap = argparse.ArgumentParser(description="카메라 프레임률 원인 진단")
    ap.add_argument("--seconds", type=float, default=5.0, help="조건당 측정 시간")
    ap.add_argument("--device", type=int, default=0, help="카메라 번호")
    ap.add_argument("--expo", action="store_true",
                    help="노출값을 훑어 FPS/밝기 맞바꿈을 본다 (이것만 실행)")
    a = ap.parse_args()
    D, S = a.device, a.seconds

    if a.expo:
        expo_sweep(D, S)
        return 0

    show_v4l2(D)

    MJPG = cv2.VideoWriter_fourcc(*"MJPG")
    YUYV = cv2.VideoWriter_fourcc(*"YUYV")

    def cfg(fourcc=None, w=None, h=None, fps=None, expo=None, buf=1):
        def f(c):
            # 형식을 먼저, 그다음 해상도, 마지막에 프레임률. 순서가 중요하다.
            if fourcc is not None:
                c.set(cv2.CAP_PROP_FOURCC, fourcc)
            if w is not None:
                c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if fps is not None:
                c.set(cv2.CAP_PROP_FPS, fps)
            if expo is not None:
                c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = 수동 (드라이버마다 다름)
                c.set(cv2.CAP_PROP_EXPOSURE, expo)
            c.set(cv2.CAP_PROP_BUFFERSIZE, buf)
        return f

    print("=" * 72)
    print(" [1] 조건별 실측  (각 %.0f초, 읽기만 반복)" % S)
    print("=" * 72)

    R = []
    R.append(trial("아무 설정 안 함", D, cfg(), S))
    R.append(trial("지금 우리 설정", D, cfg(MJPG, 640, 480, 30), S))
    R.append(trial("MJPG 640x480 (fps요청X)", D, cfg(MJPG, 640, 480), S))
    R.append(trial("YUYV 640x480", D, cfg(YUYV, 640, 480, 30), S))
    R.append(trial("MJPG 320x240", D, cfg(MJPG, 320, 240, 30), S))
    R.append(trial("MJPG 640x480 노출고정", D, cfg(MJPG, 640, 480, 30, expo=50), S))
    R.append(trial("MJPG 640x480 버퍼3", D, cfg(MJPG, 640, 480, 30, buf=3), S))
    R.append(threaded_trial("MJPG 640x480 읽기스레드", D, S))
    R = [r for r in R if r]

    if not R:
        print("측정된 조건이 없습니다.")
        return 1

    # ── 요약 ──
    print()
    print("=" * 72)
    print(" [2] 요약")
    print("=" * 72)
    print("  %-26s %8s %8s %10s" % ("조건", "실측FPS", "형식", "간격흔들림"))
    print("  " + "-" * 62)
    for r in R:
        print("  %-26s %8.1f %8s %10s"
              % (r["label"], r["fps"], r["fourcc"],
                 "%.0f~%.0f ms" % (r["lo"], r["hi"])))

    best = max(R, key=lambda r: r["fps"])
    ours = next((r for r in R if r["label"] == "지금 우리 설정"), None)

    # ── 판정 ──
    print()
    print("=" * 72)
    print(" [3] 판정")
    print("=" * 72)

    if ours:
        want = (ours["fourcc"] == "MJPG" and ours["w"] == 640 and ours["h"] == 480)
        if want:
            print("  ① cap.set() : 먹혔다. MJPG 640x480 으로 되읽혔다.")
        else:
            print("  ① cap.set() : **무시됐다.** 요청은 MJPG 640x480 인데")
            print("       실제로는 %s %dx%d 다. 이게 원인일 수 있다."
                  % (ours["fourcc"], ours["w"], ours["h"]))

        흔들림 = ours["hi"] - ours["lo"]
        if 흔들림 > ours["avg"] * 0.5:
            print("  ② 자동 노출 : 프레임 간격이 %.0f~%.0f ms 로 많이 흔들린다."
                  % (ours["lo"], ours["hi"]))
            print("       노출 시간이 장면 밝기에 따라 변하고 있을 가능성이 높다.")
        else:
            print("  ② 자동 노출 : 간격이 %.0f~%.0f ms 로 고르다. 노출 탓은 아닌 듯하다."
                  % (ours["lo"], ours["hi"]))
            print("       (고른 간격 = 드라이버가 정해진 주기로 주고 있다는 뜻)")

    print("  ③ 가장 빠른 조건 : %s  %.1f FPS" % (best["label"], best["fps"]))
    if best["fps"] > 25:
        print("       -> 카메라는 더 낼 수 있다. 이 설정을 detect_baby.py 와")
        print("          bench_models.py 에 넣으면 실제 처리량이 올라간다.")
    else:
        print("       -> 어떤 설정으로도 %.0f FPS 근처가 한계다." % best["fps"])
        print("          [0] 의 v4l2-ctl 표에서 640x480 MJPG 칸을 보라.")
        print("          거기에 30 fps 로 적혀 있는데 실측이 20 이면 노출 또는 조명이고,")
        print("          20 fps 로 적혀 있으면 이 웹캠 자체의 상한이다.")

    lowres = next((r for r in R if r["label"] == "MJPG 320x240"), None)
    if lowres and ours and lowres["fps"] > ours["fps"] * 1.3:
        print("  ④ USB 대역폭 : 해상도를 낮추니 %.1f -> %.1f FPS 로 올랐다."
              % (ours["fps"], lowres["fps"]))
        print("       대역폭 또는 센서 읽기 시간이 걸리고 있다.")
    elif lowres and ours:
        print("  ④ USB 대역폭 : 320x240 으로 낮춰도 %.1f FPS 라 별 차이가 없다."
              % lowres["fps"])
        print("       대역폭 문제는 아니다.")

    print()
    print("  ※ 조명을 밝게 켜고 한 번 더 돌려 보라. 숫자가 올라가면 노출이 원인이다.")
    print("  ※ 노출값을 훑어 어디가 최적인지 보려면 :  python3 camera_check.py --expo")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
