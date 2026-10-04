# -*- coding: utf-8 -*-
"""
보드에서 tflite 모델 3종의 속도를 자동으로 재서 표로 출력한다.

재는 것
    한 프레임 전체 시간과 그 안의 단계별 시간 (카메라 / 전처리 / 추론 / 후처리),
    유효 FPS, CPU 사용률, CPU 온도, 모델 크기.

이 표가 그대로 보고서의 "정확도-크기-속도" 3축 표의 속도 부분이 된다.
정확도와 크기는 Colab 에서 이미 나왔으므로, 여기서 나온 FPS 만 합치면 된다.

실행
    source ~/work/env/bin/activate
    cd ~/work/baby
    python3 bench_models.py                       # 폴더의 tflite 전부, 각 200프레임
    python3 bench_models.py --frames 300          # 프레임 수 지정
    python3 bench_models.py --threads 1           # 추론 스레드 수 (1/2/4 비교용)
    python3 bench_models.py --no-camera           # 카메라 없이 추론 시간만
    python3 bench_models.py best_int8.tflite      # 특정 모델만

결과
    화면에 표 + bench_result.csv 저장
"""
import os
import sys
import csv
import glob
import time
import argparse

import numpy as np
import cv2

# 윈도우 콘솔에서도 한글·표 문자가 깨지지 않게 (리눅스에서는 영향 없음)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# 추론 엔진 — 있는 것을 순서대로 쓴다 (보드: ai_edge_litert / 노트북: tensorflow.lite)
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow as _tf
        tflite = _tf.lite

try:
    import psutil
except ImportError:
    psutil = None

IMG_SIZE = 320
CONF_TH = 0.40
IOU_TH = 0.45

# 카메라 버퍼 크기 — 2026-09-23 측정으로 정했다.
#   1 로 두면 한 장을 읽어 처리하는 동안 도착한 다음 프레임이 버려진다.
#   실측 : 버퍼1 -> 16.8~20.0 FPS (간격 36~68ms 들쭉날쭉)
#          버퍼3 -> 30.0 FPS      (간격 32~36ms 일정)
#   읽기 스레드를 써도 버퍼가 1이면 15 FPS 였으므로 버퍼가 원인이다.
#   대신 프레임이 최대 3장까지 줄을 서므로 지연이 조금 늘 수 있다
#   (추론이 33ms 보다 빠르면 줄이 쌓이지 않아 실제로는 거의 없다).
CAM_BUFFER = 3



# ─────────────────────────────────────────────────────────────
def cpu_temp():
    """CPU 온도(℃). 못 읽으면 None."""
    for p in ("/sys/class/thermal/thermal_zone0/temp",):
        try:
            with open(p) as f:
                return int(f.read().strip()) / 1000.0
        except OSError:
            pass
    return None


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh))
    px, py = (size - nw) // 2, (size - nh) // 2
    out = cv2.copyMakeBorder(resized, py, size - nh - py, px, size - nw - px,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, px, py


class Runner:
    """모델 하나를 로딩하고 한 프레임 처리 시간을 단계별로 재는 것."""

    def __init__(self, path, threads):
        self.path = path
        self.size_mb = os.path.getsize(path) / 1048576.0
        try:
            self.it = tflite.Interpreter(model_path=path, num_threads=threads)
        except TypeError:
            self.it = tflite.Interpreter(model_path=path)
        self.it.allocate_tensors()
        inp = self.it.get_input_details()[0]
        out = self.it.get_output_details()[0]
        self.i_idx, self.o_idx = inp["index"], out["index"]
        self.i_dtype = inp["dtype"]
        shape = list(inp["shape"])
        self.nchw = (shape[1] == 3)
        self.size = shape[2] if self.nchw else shape[1]
        self.i_q = inp.get("quantization", (0.0, 0))
        self.o_q = out.get("quantization", (0.0, 0))
        self.i_int = inp["dtype"] in (np.int8, np.uint8)
        self.o_int = out["dtype"] in (np.int8, np.uint8)
        self.in_shape = shape
        self.out_shape = list(out["shape"])

    def step(self, frame):
        """전처리 / 추론 / 후처리 시간(ms)을 각각 돌려준다."""
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb, r, px, py = letterbox(rgb, self.size)
        if self.i_int:
            s, z = self.i_q[0], self.i_q[1]
            x = lb.astype(np.float32) / 255.0 / (s if s else 1.0) + z
            x = np.clip(x, -128, 127).astype(self.i_dtype)
        else:
            x = lb.astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        if self.nchw:
            x = np.transpose(x, (0, 3, 1, 2))
        t1 = time.perf_counter()

        self.it.set_tensor(self.i_idx, x)
        self.it.invoke()
        raw = self.it.get_tensor(self.o_idx)[0]
        t2 = time.perf_counter()

        if self.o_int:
            s, z = self.o_q[0], self.o_q[1]
            raw = (raw.astype(np.float32) - z) * s
        raw = raw.transpose()
        scores = raw[:, 4:]
        conf = scores.max(axis=1)
        cls = scores.argmax(axis=1)
        m = conf > CONF_TH
        n_det = 0
        if m.any():
            f = raw[m]
            cx, cy, w, h = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=-1)
            keep = cv2.dnn.NMSBoxesBatched(boxes, conf[m], cls[m],
                                           score_threshold=CONF_TH,
                                           nms_threshold=IOU_TH)
            n_det = len(np.array(keep).flatten())
        t3 = time.perf_counter()

        return ((t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, n_det)


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="tflite 모델 속도 자동 측정")
    ap.add_argument("models", nargs="*", help="측정할 tflite (없으면 폴더 전체)")
    ap.add_argument("--frames", type=int, default=200, help="모델당 프레임 수")
    ap.add_argument("--threads", type=int, default=4, help="추론 스레드 수")
    ap.add_argument("--no-camera", action="store_true", help="카메라 없이 추론만")
    ap.add_argument("--out", default="bench_result.csv")
    ap.add_argument("--peek", action="store_true",
                    help="카메라로 찍힌 한 장을 bench_peek.jpg 로 저장 (확인용)")
    a = ap.parse_args()

    models = a.models or sorted(glob.glob("*.tflite"))
    if not models:
        print("tflite 파일이 없습니다.")
        return 1

    cap = None
    if not a.no_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAM_BUFFER)
        if not cap.isOpened():
            print("[경고] 카메라를 못 열어서 고정 이미지로 측정합니다.")
            cap = None
    dummy = np.full((480, 640, 3), 120, np.uint8)

    # 카메라가 정말로 찍히고 있는지 눈으로 확인하고 싶을 때.
    # 이 도구는 속도 측정용이라 창을 띄우지 않는다(그리는 시간이 측정에 섞인다).
    if a.peek and cap:
        for _ in range(5):          # 자동 노출이 잡힐 시간
            cap.read()
        ok, fr = cap.read()
        if ok:
            cv2.imwrite("bench_peek.jpg", fr)
            print("  [확인] 지금 카메라에 찍힌 화면 : bench_peek.jpg")
        else:
            print("  [확인] 카메라에서 한 장도 못 읽었습니다.")

    print("=" * 74)
    print(" 모델 %d개 / 각 %d프레임 / 추론 스레드 %d / 카메라 %s"
          % (len(models), a.frames, a.threads, "사용" if cap else "미사용"))
    print("=" * 74)

    rows = []
    for path in models:
        try:
            rn = Runner(path, a.threads)
        except Exception as e:
            print("\n[%s] 로딩 실패 : %s" % (path, e))
            continue

        print("\n[%s]  %.2f MB  입력 %s %s  출력 %s"
              % (os.path.basename(path), rn.size_mb, rn.in_shape,
                 "NCHW" if rn.nchw else "NHWC", rn.out_shape))

        # 워밍업 — 처음 몇 번은 유독 느리므로 측정에서 뺀다
        for _ in range(5):
            ok, fr = (cap.read() if cap else (True, dummy))
            rn.step(fr if ok else dummy)

        t_cam, t_pre, t_inf, t_post = [], [], [], []
        n_det_tot = 0
        if psutil:
            psutil.cpu_percent(interval=None)
        temp0 = cpu_temp()
        t_start = time.time()

        for i in range(a.frames):
            tc = time.perf_counter()
            if cap:
                ok, fr = cap.read()
                if not ok:
                    fr = dummy
            else:
                fr = dummy
            t_cam.append((time.perf_counter() - tc) * 1000.0)
            pre, inf, post, nd = rn.step(fr)
            t_pre.append(pre)
            t_inf.append(inf)
            t_post.append(post)
            n_det_tot += nd
            if (i + 1) % 50 == 0:
                print("   %d/%d ..." % (i + 1, a.frames))

        elapsed = time.time() - t_start
        cpu = psutil.cpu_percent(interval=None) if psutil else float("nan")
        temp1 = cpu_temp()

        def avg(x):
            return sum(x) / len(x) if x else 0.0

        frame_ms = avg(t_cam) + avg(t_pre) + avg(t_inf) + avg(t_post)
        row = {
            "모델": os.path.basename(path),
            "크기(MB)": round(rn.size_mb, 2),
            "카메라(ms)": round(avg(t_cam), 2),
            "전처리(ms)": round(avg(t_pre), 2),
            "추론(ms)": round(avg(t_inf), 2),
            "후처리(ms)": round(avg(t_post), 2),
            "한프레임(ms)": round(frame_ms, 2),
            "FPS(계산)": round(1000.0 / frame_ms, 1) if frame_ms else 0,
            "FPS(실측)": round(a.frames / elapsed, 1),
            "CPU(%)": round(cpu, 1),
            "온도(C)": round(temp1, 1) if temp1 else "",
            "검출/프레임": round(n_det_tot / a.frames, 2),
        }
        rows.append(row)
        print("   한 프레임 %.2f ms  ->  %.1f FPS   (추론 %.2f ms)"
              % (frame_ms, row["FPS(계산)"], avg(t_inf)))

    if cap:
        cap.release()

    if not rows:
        return 1

    # ── 표 출력 ──
    keys = list(rows[0].keys())
    w = {k: max(len(str(k)), max(len(str(r[k])) for r in rows)) for k in keys}
    print("\n" + "=" * 74)
    print(" 결과")
    print("=" * 74)
    print("  ".join(str(k).rjust(w[k]) for k in keys))
    print("-" * (sum(w.values()) + 2 * (len(keys) - 1)))
    for r in rows:
        print("  ".join(str(r[k]).rjust(w[k]) for k in keys))

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("\n저장 : %s" % os.path.abspath(a.out))

    # ── 해석 도움말 ──
    print("\n[ 읽는 법 ]")
    print("  FPS(계산)  단계별 시간을 더해 뒤집은 값. 파이프라인 자체의 한계")
    print("  FPS(실측)  실제 걸린 시간 기준. 카메라가 못 따라오면 이쪽이 낮게 나온다")
    print("  두 값이 크게 다르면 카메라가 병목이다 (camera_check.py 로 확인)")
    print("  추론(ms) 가 가장 큰 항목이면 모델 경량화가 효과를 낸다")
    print("  후처리(ms) 가 크면 NMS 가 병목 — 검출 개수가 많을 때 그렇다")
    print()
    print("  카메라(ms) 가 0.0 이면 카메라를 못 열어 고정 이미지로 잰 것이다.")
    print("  0 이 아니면 실제로 카메라를 읽고 있다는 뜻이다.")
    print("  찍힌 화면을 눈으로 보려면 --peek 을 붙여라 (bench_peek.jpg 로 저장)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
