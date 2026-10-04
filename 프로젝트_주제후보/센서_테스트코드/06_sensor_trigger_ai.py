#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06. 센서 트리거 -> AI 기동 (후보 H 본체) + 비교 측정 하네스.

두 가지 방식을 같은 조건에서 돌려 CSV로 남긴다.

    always   : AI를 쉬지 않고 계속 돌린다            (기존 방식 = Before)
    trigger  : 센서가 감지했을 때만 AI를 깨운다       (제안 방식 = After)

기록: 평균/최대 CPU 사용률, 추론 횟수, AI 가동 시간 비율,
      트리거 -> 판정 완료 지연(ms), 이벤트 수.
이 CSV가 그대로 보고서의 Before/After 비교표가 된다.

사용 예
-------
    # 센서 없이 먼저 동작만 확인 (버튼을 트리거로 사용)
    python3 06_sensor_trigger_ai.py --mode trigger --sensor button --sec 60

    # 실제 측정 (PIR 센서, 각 2분)
    python3 06_sensor_trigger_ai.py --mode always  --sec 120
    python3 06_sensor_trigger_ai.py --mode trigger --sec 120

    # 실제 모델과 카메라를 물려서
    python3 06_sensor_trigger_ai.py --mode trigger --model best_int8.tflite --camera

주의: 두 모드를 같은 조명/같은 활동량에서 재야 비교가 성립한다.
      한 사람이 같은 리듬으로 움직이면서 두 번 돌리는 것이 가장 낫다.
"""
import argparse, csv, os, sys, threading, time

# ── 선택 의존성 ──────────────────────────────────────────────
try:
    import psutil
except ImportError:
    psutil = None

try:
    import numpy as np
except ImportError:
    np = None

SIM_N = 220          # 모의 부하 크기. 보드 실측에 맞춰 조정한다


# ── 1. 추론부 ────────────────────────────────────────────────
class Inference:
    """실제 tflite 모델이 있으면 그것으로, 없으면 비슷한 부하의 대역 연산으로 돌린다."""

    def __init__(self, model_path=None, use_camera=False):
        self.kind = "simulated"
        self.interp = None
        self.cap = None

        if model_path and os.path.exists(model_path):
            try:
                try:
                    from ai_edge_litert.interpreter import Interpreter
                except ImportError:
                    from tflite_runtime.interpreter import Interpreter
                self.interp = Interpreter(model_path=model_path, num_threads=4)
                self.interp.allocate_tensors()
                self.inp = self.interp.get_input_details()[0]
                self.out = self.interp.get_output_details()[0]
                self.kind = "tflite:" + os.path.basename(model_path)
            except Exception as e:
                print("[경고] 모델 로딩 실패 -> 모의 연산으로 대체:", e)

        if use_camera:
            try:
                import cv2
                self.cv2 = cv2
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                if not self.cap.isOpened():
                    print("[경고] 카메라 열기 실패 -> 카메라 없이 진행")
                    self.cap = None
            except ImportError:
                print("[경고] opencv 없음 -> 카메라 없이 진행")

    def run_once(self):
        """한 번 추론하고 걸린 시간(ms)을 돌려준다."""
        t0 = time.perf_counter()

        if self.cap is not None:
            self.cap.read()

        if self.interp is not None:
            shape = self.inp["shape"]
            dtype = self.inp["dtype"]
            if np is not None:
                if dtype.__name__.startswith("float"):
                    x = np.zeros(shape, dtype=dtype)
                else:
                    x = np.zeros(shape, dtype=dtype)
                self.interp.set_tensor(self.inp["index"], x)
                self.interp.invoke()
                self.interp.get_tensor(self.out["index"])
        else:
            # 모의 부하. 실제 모델이 없을 때 쓰는 대역이다.
            # SIM_N을 조정해 보드에서의 실제 추론 시간과 비슷하게 맞춘 뒤 측정할 것
            # (부하 크기가 다르면 always/trigger 비교 자체는 성립하지만,
            #  절대 수치를 실제 파이프라인 값으로 말할 수 없다).
            if np is not None:
                a = np.random.rand(SIM_N, SIM_N).astype("float32")
                for _ in range(3):
                    a = a @ a.T
                    a /= (a.max() + 1e-6)
            else:
                s = 0.0
                for i in range(400000):
                    s += i * 0.5

        return (time.perf_counter() - t0) * 1000.0

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ── 2. 센서부 ────────────────────────────────────────────────
class Trigger:
    """감지 여부를 bool로 알려주는 얇은 껍데기. 센서가 없으면 버튼이나 주기 신호로 대체."""

    PIR_PIN = 22
    BTN_PIN = 27

    def __init__(self, kind):
        self.kind = kind
        self.dev = None
        if kind == "pir":
            from gpiozero import MotionSensor
            self.dev = MotionSensor(self.PIR_PIN)
            print("PIR 안정화 대기 30초...")
            time.sleep(30)
        elif kind == "button":
            from gpiozero import Button
            self.dev = Button(self.BTN_PIN, pull_up=True, bounce_time=0.05)
        elif kind == "none":
            pass
        else:
            raise ValueError("알 수 없는 센서 종류: %s" % kind)

    @property
    def active(self):
        if self.kind == "pir":
            return bool(self.dev.motion_detected)
        if self.kind == "button":
            return bool(self.dev.is_pressed)
        # none: 10초 주기로 2초간 켜지는 가짜 신호 (배선 전 동작 확인용)
        return (time.time() % 10.0) < 2.0

    def close(self):
        if self.dev is not None:
            self.dev.close()


# ── 3. CPU 사용률 샘플러 ─────────────────────────────────────
class CpuSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._done = threading.Event()

    def run(self):
        if psutil is None:
            return
        psutil.cpu_percent(interval=None)          # 첫 호출은 버린다
        while not self._done.is_set():
            self.samples.append(psutil.cpu_percent(interval=self.interval))

    def stop(self):
        self._done.set()

    def summary(self):
        if not self.samples:
            return (float("nan"), float("nan"))
        return (sum(self.samples) / len(self.samples), max(self.samples))


# ── 4. 본체 ──────────────────────────────────────────────────
def run(args):
    infer = Inference(args.model, args.camera)
    trig = Trigger(args.sensor if args.mode == "trigger" else "none")
    print("\n모드: %s  |  추론: %s  |  센서: %s  |  측정 %d초"
          % (args.mode, infer.kind, trig.kind, args.sec))
    print("측정을 시작합니다. 두 모드에서 같은 리듬으로 움직여 주세요.\n")

    sampler = CpuSampler()
    sampler.start()

    rows = []
    t_start = time.time()
    t_end = t_start + args.sec
    n_infer = 0
    busy_ms = 0.0
    n_events = 0
    prev_active = False
    t_event = None

    try:
        while time.time() < t_end:
            now = time.time()

            if args.mode == "always":
                ms = infer.run_once()
                n_infer += 1
                busy_ms += ms
                rows.append([round(now - t_start, 3), "infer", round(ms, 2), ""])

            else:  # trigger
                active = trig.active
                if active and not prev_active:          # 감지 시작 순간
                    n_events += 1
                    t_event = time.perf_counter()
                prev_active = active

                if active:
                    ms = infer.run_once()
                    n_infer += 1
                    busy_ms += ms
                    latency = ""
                    if t_event is not None:             # 트리거 -> 첫 판정 완료까지
                        latency = round((time.perf_counter() - t_event) * 1000.0, 2)
                        t_event = None
                    rows.append([round(now - t_start, 3), "infer", round(ms, 2), latency])
                else:
                    time.sleep(0.02)                    # AI 정지 구간
    except KeyboardInterrupt:
        print("\n중단됨 — 여기까지의 결과를 저장합니다.")

    elapsed = time.time() - t_start
    sampler.stop()
    sampler.join(timeout=1.0)
    cpu_avg, cpu_max = sampler.summary()
    infer.close()
    trig.close()

    # ── 결과 저장 ──
    out = args.out or ("result_%s.csv" % args.mode)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# mode", args.mode])
        w.writerow(["# inference", infer.kind])
        w.writerow(["# sensor", trig.kind])
        w.writerow(["# elapsed_s", round(elapsed, 2)])
        w.writerow(["# infer_count", n_infer])
        w.writerow(["# ai_busy_ratio_pct", round(busy_ms / 10.0 / elapsed, 2)])
        w.writerow(["# cpu_avg_pct", round(cpu_avg, 2)])
        w.writerow(["# cpu_max_pct", round(cpu_max, 2)])
        w.writerow(["# events", n_events])
        w.writerow([])
        w.writerow(["t_sec", "kind", "infer_ms", "trigger_latency_ms"])
        w.writerows(rows)

    # ── 화면 요약 ──
    lat = [r[3] for r in rows if r[3] != ""]
    print("\n" + "=" * 52)
    print(" 결과  (%s)" % args.mode)
    print("=" * 52)
    print("  측정 시간          : %.1f 초" % elapsed)
    print("  추론 횟수          : %d 회" % n_infer)
    print("  AI 가동 시간 비율  : %.1f %%" % (busy_ms / 10.0 / elapsed))
    print("  평균 추론 시간     : %.1f ms" % (busy_ms / n_infer if n_infer else 0))
    if psutil is None:
        print("  CPU 사용률         : psutil 미설치 (pip install psutil)")
    else:
        print("  CPU 사용률         : 평균 %.1f %% / 최대 %.1f %%" % (cpu_avg, cpu_max))
    if args.mode == "trigger":
        print("  트리거 이벤트      : %d 회" % n_events)
        if lat:
            print("  트리거->판정 지연  : 평균 %.1f ms / 최대 %.1f ms"
                  % (sum(lat) / len(lat), max(lat)))
    print("\n  저장: %s" % out)
    print("\n  두 모드를 모두 돌린 뒤 result_always.csv 와 result_trigger.csv 의")
    print("  머리말(# 로 시작하는 줄)을 나란히 놓으면 그것이 Before/After 표다.")


def main():
    p = argparse.ArgumentParser(description="센서 트리거 -> AI 기동 비교 측정")
    p.add_argument("--mode", choices=["always", "trigger"], required=True,
                   help="always=상시 구동(Before), trigger=센서 감지 시에만(After)")
    p.add_argument("--sensor", choices=["pir", "button", "none"], default="pir",
                   help="트리거 소스. 배선 전이면 button 또는 none으로 먼저 확인")
    p.add_argument("--sec", type=int, default=120, help="측정 시간(초)")
    p.add_argument("--model", default=None, help="tflite 모델 경로 (없으면 모의 연산)")
    p.add_argument("--camera", action="store_true", help="실제 카메라 캡처를 포함")
    p.add_argument("--out", default=None, help="CSV 저장 경로")
    args = p.parse_args()

    if psutil is None:
        print("[안내] psutil이 없어 CPU 사용률을 못 잽니다.  pip install psutil\n")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
