# -*- coding: utf-8 -*-
"""
실사용 검출률 비교 — 같은 장면으로 모델 여러 개를 한 번에 잰다.

왜 필요한가
    Colab 의 mAP 는 "학습 사진과 같은 날 같은 자리에서 찍은" 시험지로 잰 값이다.
    그래서 mAP50 0.98 이 나와도 실제 웹캠 앞에서는 칼을 못 찾을 수 있다.
    **실제로 카메라에 대고 재는 수치**가 따로 필요하다.

어떻게 공정하게 재나
    모델마다 따로 촬영하면 손 위치·조명·각도가 달라져 비교가 무의미해진다.
    그래서 **프레임을 한 번만 찍어 메모리에 담아 두고, 그 똑같은 사진들을
    모든 모델에 그대로 먹인다.** 장면 차이가 0이 되므로 순수하게 모델만 비교된다.

        [촬영 1회] ──┬──▶ 모델 A 에 전부 먹임
                     ├──▶ 모델 B 에 전부 먹임
                     └──▶ 모델 C 에 전부 먹임

실행
    source ~/work/env/bin/activate
    cd ~/work/baby

    # 1) 칼을 카메라 앞에 놓고 (인형도 같이) 가만히 둔 다음
    python3 detect_rate.py

    # 다른 사용법
    python3 detect_rate.py --frames 60          # 찍을 장수 (기본 60)
    python3 detect_rate.py --conf 0.25,0.40,0.50   # 여러 문턱값을 한 장면으로 비교
    python3 detect_rate.py --save               # 찍은 장면과 결과 그림 저장
    python3 detect_rate.py a.tflite b.tflite    # 특정 모델만

읽는 법
    검출률   그 클래스가 몇 %의 프레임에서 잡혔는가.  놓치면 사고이므로 이게 핵심
    평균신뢰 잡았을 때 얼마나 확신했는가. 낮으면 문턱값을 조금만 올려도 놓친다
"""
import os
import sys
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

# 카메라 버퍼 크기 — 2026-09-23 측정으로 정했다.
#   1 로 두면 한 장을 읽어 처리하는 동안 도착한 다음 프레임이 버려진다.
#   실측 : 버퍼1 -> 16.8~20.0 FPS (간격 36~68ms 들쭉날쭉)
#          버퍼3 -> 30.0 FPS      (간격 32~36ms 일정)
#   읽기 스레드를 써도 버퍼가 1이면 15 FPS 였으므로 버퍼가 원인이다.
#   대신 프레임이 최대 3장까지 줄을 서므로 지연이 조금 늘 수 있다
#   (추론이 33ms 보다 빠르면 줄이 쌓이지 않아 실제로는 거의 없다).
CAM_BUFFER = 3

NAMES = {0: "baby", 1: "adult", 2: "knife", 3: "outlet"}
COLORS = {0: (0, 200, 255), 1: (200, 200, 200), 2: (0, 0, 255), 3: (0, 0, 255)}
IOU_TH = 0.45


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(w * r), int(h * r)
    out = cv2.copyMakeBorder(cv2.resize(img, (nw, nh)),
                             (size - nh) // 2, size - nh - (size - nh) // 2,
                             (size - nw) // 2, size - nw - (size - nw) // 2,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (size - nw) // 2, (size - nh) // 2


class Model:
    def __init__(self, path, threads=4):
        self.path = path
        self.name = os.path.basename(path)
        self.size_mb = os.path.getsize(path) / 1048576.0
        try:
            self.it = tflite.Interpreter(model_path=path, num_threads=threads)
        except TypeError:
            self.it = tflite.Interpreter(model_path=path)
        self.it.allocate_tensors()
        inp, out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.i_idx, self.o_idx = inp["index"], out["index"]
        self.i_dtype = inp["dtype"]
        shape = list(inp["shape"])
        self.nchw = (shape[1] == 3)
        self.size = shape[2] if self.nchw else shape[1]
        self.i_q, self.o_q = inp.get("quantization", (0.0, 0)), out.get("quantization", (0.0, 0))
        self.i_int = inp["dtype"] in (np.int8, np.uint8)
        self.o_int = out["dtype"] in (np.int8, np.uint8)
        # 입출력 dtype 만으로는 내부가 INT8 인지 알 수 없다.
        #   Ultralytics 가 내보낸 INT8 모델은 내부 연산만 8비트이고
        #   입출력에는 Quantize/Dequantize 노드를 끼워 float32 를 유지한다.
        #   그래서 여기서는 '입출력이 무엇인가'만 적고, 실제 INT8 여부는
        #   파일 크기와 추론 시간으로 판단한다(아래 표의 안내 참고).
        self.kind = "int8 I/O" if self.i_int else "float I/O"

    def detect(self, frame, conf_th):
        """conf_th 이상인 검출을 돌려준다. 여러 문턱값을 볼 때는
        가장 낮은 값으로 한 번만 부르고, 점수로 다시 거르면 된다."""
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

        self.it.set_tensor(self.i_idx, x)
        self.it.invoke()
        raw = self.it.get_tensor(self.o_idx)[0]
        if self.o_int:
            s, z = self.o_q[0], self.o_q[1]
            raw = (raw.astype(np.float32) - z) * s
        raw = raw.transpose()

        scores = raw[:, 4:]
        conf, cls = scores.max(axis=1), scores.argmax(axis=1)
        m = conf > conf_th
        if not m.any():
            return []
        f = raw[m]
        sc, cl = conf[m], cls[m]
        cx, cy, w, h = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=-1)
        keep = cv2.dnn.NMSBoxesBatched(boxes, sc, cl,
                                       score_threshold=conf_th, nms_threshold=IOU_TH)
        res = []
        H, W = frame.shape[:2]
        for i in np.array(keep).flatten():
            bx, by, bw, bh = boxes[i] * self.size
            res.append((int(np.clip((bx - px) / r, 0, W)),
                        int(np.clip((by - py) / r, 0, H)),
                        int(np.clip((bx + bw - px) / r, 0, W)),
                        int(np.clip((by + bh - py) / r, 0, H)),
                        float(sc[i]), int(cl[i])))
        return res


def main():
    ap = argparse.ArgumentParser(description="같은 장면으로 모델들의 실사용 검출률 비교")
    ap.add_argument("models", nargs="*", help="비교할 tflite (없으면 폴더 전체)")
    ap.add_argument("--frames", type=int, default=60, help="찍을 장수")
    ap.add_argument("--conf", default="0.25,0.40,0.50",
                    help="신뢰도 문턱값. 쉼표로 여러 개 (기본 0.25,0.40,0.50)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save", action="store_true", help="모델별 결과 그림도 저장")
    ap.add_argument("--no-gui", action="store_true",
                    help="미리보기 창 없이 3초 뒤 바로 촬영")
    ap.add_argument("--no-mirror", action="store_true",
                    help="좌우 반전 끄기 (기본은 거울처럼 반전)")
    ap.add_argument("--wait", type=float, default=8.0,
                    help="촬영 전 준비 시간(초). 기본 8초")
    ap.add_argument("--out", default="detect_rate.csv")
    a = ap.parse_args()
    a.no_gui = getattr(a, "no_gui", False)
    try:
        THS = sorted({round(float(x), 3) for x in str(a.conf).split(",") if x.strip()})
    except ValueError:
        print("--conf 는 숫자 또는 쉼표로 이은 숫자여야 합니다. 예: --conf 0.25,0.40")
        return 1
    if not THS:
        THS = [0.40]
    TH_MIN, TH_MAIN = THS[0], (0.40 if 0.40 in THS else THS[-1])

    paths = a.models or sorted(glob.glob("*.tflite"))
    if not paths:
        print("tflite 파일이 없습니다.")
        return 1

    # ── 1) 장면을 한 번만 찍는다 ──
    print("=" * 74)
    print(" [1] 장면 촬영 — 비교할 물체를 카메라 앞에 두고 가만히 있으세요")
    print("=" * 74)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAM_BUFFER)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return 1
    gui = not a.no_gui
    if gui:
        # 키 입력에 의존하지 않는다. X11 로 접속했을 때 키가 창에 전달되지 않아
        # 준비도 되기 전에 촬영이 시작되는 일이 있었다.
        # 그래서 **정해진 시간 동안 화면을 보여 주고 자동으로** 시작한다.
        # 스페이스를 누르면 빨리 시작하고, q 를 누르면 취소된다(눌리면).
        print("   화면을 보고 물체가 다 들어왔는지 확인하세요.")
        print("   %.0f초 뒤 자동으로 촬영합니다. (스페이스=바로 시작, q=취소)" % a.wait)
        try:
            cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
            t_end = time.time() + a.wait
            while True:
                left = t_end - time.time()
                if left <= 0:
                    break
                ok, f = cap.read()
                if not ok:
                    continue
                if not a.no_mirror:
                    f = cv2.flip(f, 1)
                view = f.copy()
                cv2.putText(view, "%.0f" % (left + 0.99), (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 4)
                cv2.putText(view, "촬영까지".encode("ascii", "ignore").decode() or "",
                            (20, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
                cv2.putText(view, "starting soon...  SPACE=now  q=cancel",
                            (20, 110), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)
                cv2.imshow("scene", view)
                k = cv2.waitKey(30) & 0xFF
                if k == ord(" "):
                    break
                if k == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("   취소했습니다.")
                    return 1
        except cv2.error as e:
            print("   (창을 띄울 수 없어 미리보기를 건너뜁니다: %s)" % e)
            gui = False

    if not gui:
        for i in range(int(a.wait), 0, -1):      # 자세 잡을 시간
            print("   %d..." % i)
            t0 = time.time()
            while time.time() - t0 < 1.0:
                cap.read()

    print("   촬영 중 — 움직이지 마세요")
    frames = []
    while len(frames) < a.frames:
        ok, f = cap.read()
        if ok:
            if not a.no_mirror:
                f = cv2.flip(f, 1)          # 본 화면과 같게 (거울)
            frames.append(f.copy())
            if gui:
                v = f.copy()
                cv2.putText(v, "REC %d/%d" % (len(frames), a.frames), (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                try:
                    cv2.imshow("scene", v)
                    cv2.waitKey(1)
                except cv2.error:
                    pass
            if len(frames) % 20 == 0:
                print("   %d/%d" % (len(frames), a.frames))
    cap.release()
    if gui:
        cv2.destroyAllWindows()
    print("   촬영 끝. 이 %d장을 모든 모델에 똑같이 먹입니다." % len(frames))

    # 찍힌 장면은 항상 남긴다. "왜 0% 인가"를 따질 때 이것부터 봐야 한다.
    os.makedirs("rate_out", exist_ok=True)
    cv2.imwrite("rate_out/scene.jpg", frames[len(frames) // 2])
    print("   찍힌 장면 : rate_out/scene.jpg  ← 결과가 이상하면 이것부터 확인")

    # ── 2) 모델마다 같은 사진으로 ──
    print()
    print("=" * 74)
    print(" [2] 모델별 검출  (문턱값 %s — 추론은 한 번만 하고 점수로 다시 센다)"
          % ", ".join("%.2f" % t for t in THS))
    print("=" * 74)

    rows = []
    by_th = {}          # 문턱값 -> [행, ...]
    for p in paths:
        try:
            m = Model(p, a.threads)
        except Exception as e:
            print("\n[%s] 로딩 실패 : %s" % (p, e))
            continue

        # 문턱값별 집계 그릇
        hit = {t: {c: 0 for c in NAMES} for t in THS}
        conf_sum = {t: {c: 0.0 for c in NAMES} for t in THS}
        t0 = time.time()
        mid_dets = None
        for i, f in enumerate(frames):
            dets = m.detect(f, TH_MIN)          # 가장 낮은 문턱값으로 한 번만
            if i == len(frames) // 2:
                mid_dets = [d for d in dets if d[4] >= TH_MAIN]
            for t in THS:
                seen = {}
                for d in dets:
                    if d[4] < t:
                        continue
                    c = d[5]
                    if c in hit[t]:
                        seen[c] = max(seen.get(c, 0.0), d[4])
                for c, sc in seen.items():
                    hit[t][c] += 1
                    conf_sum[t][c] += sc
        dt = time.time() - t0

        n = len(frames)
        for t in THS:
            row = {"모델": m.name, "문턱값": "%.2f" % t,
                   "입출력": m.kind, "크기(MB)": round(m.size_mb, 2)}
            for c, nm in NAMES.items():
                row[nm + "검출률"] = "%.0f%%" % (hit[t][c] / n * 100)
                row[nm + "신뢰도"] = ("%.2f" % (conf_sum[t][c] / hit[t][c])) if hit[t][c] else "-"
            row["추론(ms)"] = round(dt * 1000.0 / n, 1)
            by_th.setdefault(t, []).append(row)
            rows.append(row)

        print("  %-22s %s  추론 %.1f ms" % (m.name, m.kind, dt * 1000.0 / n))
        for t in THS:
            print("      문턱값 %.2f  ->  knife %3.0f%%   baby %3.0f%%   adult %3.0f%%"
                  % (t, hit[t][2] / n * 100, hit[t][0] / n * 100, hit[t][1] / n * 100))

        if a.save and mid_dets is not None:
            img = frames[len(frames) // 2].copy()
            for (x1, y1, x2, y2, sc, c) in mid_dets:
                col = COLORS.get(c, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                cv2.putText(img, "%s %d%%" % (NAMES.get(c, c), int(sc * 100)),
                            (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_PLAIN, 1, col, 2)
            cv2.putText(img, m.name, (10, 24), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)
            cv2.imwrite("rate_out/%s.jpg" % os.path.splitext(m.name)[0], img)

    if not rows:
        return 1

    # ── 3) 표 ──
    keys = list(rows[0].keys())
    w = {k: max(len(str(k)), max(len(str(r[k])) for r in rows)) for k in keys}
    print()
    print("=" * 74)
    print(" [3] 결과 — 같은 %d장, 문턱값별" % len(frames))
    print("=" * 74)
    for t in THS:
        print()
        print(" ── 문턱값 %.2f ──" % t)
        print("  ".join(str(k).rjust(w[k]) for k in keys))
        print("-" * (sum(w.values()) + 2 * (len(keys) - 1)))
        for r in by_th[t]:
            print("  ".join(str(r[k]).rjust(w[k]) for k in keys))

    # ── 모든 모델이 0% 인 클래스는 모델 탓이 아닐 가능성이 크다 ──
    dead = [nm for c, nm in NAMES.items()
            if all(r[nm + "검출률"] == "0%" for r in by_th[TH_MIN])]
    if dead and len(rows) > 1:
        print()
        print("!" * 74)
        print(" %s 가 **모든 모델에서 0%%** 입니다." % ", ".join(dead))
        print(" 모델이 원인이라면 이렇게 전부 똑같이 0 이 나오기 어렵습니다.")
        print(" 먼저 의심할 것 :")
        print("   1) 그 물체가 화면에 안 들어왔다  -> rate_out/scene.jpg 를 열어보세요")
        print("   2) 너무 작거나 가려졌다          -> 카메라에 더 가까이")
        print("   3) 문턱값이 높다                 -> --conf 0.25 로 다시")
        print("!" * 74)

    import csv
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("\n저장 : %s" % os.path.abspath(a.out))
    if a.save:
        print("       rate_out/ 에 장면과 모델별 결과 그림")

    print()
    print("[ 읽는 법 ]")
    print("  검출률이 핵심이다. 놓치면 사고인 시스템이므로 신뢰도보다 먼저 본다")
    print("  '입출력' 칸은 입출력 자료형일 뿐 내부 연산이 아니다.")
    print("  Ultralytics 의 INT8 모델은 내부만 8비트이고 입출력은 float 로 나온다.")
    print("  실제 INT8 여부는 **크기와 추론 시간**으로 본다 (작고 빠르면 INT8)")
    print("  같은 학습인데 FP32 는 되고 INT8 이 안 되면  -> 양자화 탓")
    print("  같은 양자화인데 모델끼리 다르면              -> 학습·데이터 탓")
    print("  문턱값을 낮췄을 때 검출률이 확 오르면 '못 찾는' 게 아니라")
    print("  '자신 없어 하는' 것이므로 detect_baby.py 의 CONF_TH 조정으로 해결된다")
    print()
    print("  [ 거리 주의 ] 물체가 작게 찍히면 어떤 모델도 못 찾는다.")
    print("  거리를 바꾼 두 측정을 비교하면 안 된다 — 모델 차이가 아니라 거리 차이다.")
    print("  거리별 성능을 보려면 같은 물체를 가까이/중간/멀리 두고 각각 재라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
