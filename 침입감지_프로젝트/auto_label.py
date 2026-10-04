# -*- coding: utf-8 -*-
"""
사진 폴더에 모델을 돌려 라벨(.txt)을 미리 만들어 둔다. 그다음 사람이 확인·수정한다.

** 먼저 알아야 할 한계 **
    자동 라벨링은 "모델이 찾은 것"을 적는 방식이다.
    그런데 우리가 지금 보강하려는 것은 **모델이 못 찾는 경우**(기울어진 칼, 먼 거리)다.
    못 찾는 것을 자동으로 라벨링할 수는 없다.
    그래서 이 도구는 "일을 대신 해 주는 것"이 아니라
    **쉬운 것을 미리 채워 두고, 사람이 손댈 곳을 알려 주는 것**이다.

        정면·근거리 칼  -> 자동으로 채워짐 (시간 절약)
        기울어진·먼 칼  -> 비어서 나옴. 여기를 직접 그려야 하고, 그게 우리가 필요한 데이터다

    문턱값을 일부러 낮게(0.25) 잡는다. 애매한 것도 일단 그려 두면
    사람이 지우거나 고치는 편이 처음부터 그리는 것보다 빠르기 때문이다.

실행 (보드에서)
    source ~/work/env/bin/activate
    cd ~/work/baby

    python3 auto_label.py captures/                     # 기본 모델로
    python3 auto_label.py captures/ --model v2_best.tflite
    python3 auto_label.py captures/ --conf 0.15         # 더 많이 잡기
    python3 auto_label.py captures/ --review            # 확인용 그림도 저장
    python3 auto_label.py captures/ --force             # 이미 있는 .txt 도 덮어쓰기

    기본적으로 **이미 .txt 가 있는 사진은 건드리지 않는다**(손으로 한 작업 보호).

만든 뒤
    폴더를 PC 로 가져와 LabelImg 로 열어 확인·수정한다.
    LabelImg 는 YOLO 형식, classes.txt 는 이 도구가 같이 만들어 둔다.
"""
import os
import sys
import glob
import argparse

import numpy as np
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

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
        self.name = os.path.basename(path)
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
        self.i_q = inp.get("quantization", (0.0, 0))
        self.o_q = out.get("quantization", (0.0, 0))
        self.i_int = inp["dtype"] in (np.int8, np.uint8)
        self.o_int = out["dtype"] in (np.int8, np.uint8)

    def detect(self, frame, conf_th):
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
        H, W = frame.shape[:2]
        res = []
        for i in np.array(keep).flatten():
            bx, by, bw, bh = boxes[i] * self.size
            x1 = float(np.clip((bx - px) / r, 0, W))
            y1 = float(np.clip((by - py) / r, 0, H))
            x2 = float(np.clip((bx + bw - px) / r, 0, W))
            y2 = float(np.clip((by + bh - py) / r, 0, H))
            if x2 - x1 < 4 or y2 - y1 < 4:      # 너무 작은 것은 버린다
                continue
            res.append((x1, y1, x2, y2, float(sc[i]), int(cl[i])))
        return res


def imread_any(path):
    """한글 경로에서도 읽히게. cv2.imread 는 한글 경로에서 실패한다."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="모델로 라벨을 미리 만들어 둔다")
    ap.add_argument("folder", help="사진(.jpg)이 들어 있는 폴더")
    ap.add_argument("--model", default="v2_best.tflite",
                    help="라벨링에 쓸 모델 (검출이 가장 좋은 것을 쓸 것)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="문턱값. 낮게 잡아야 사람이 지우기만 하면 된다 (기본 0.25)")
    ap.add_argument("--force", action="store_true",
                    help="이미 .txt 가 있어도 덮어쓴다 (기본은 건드리지 않음)")
    ap.add_argument("--review", action="store_true",
                    help="박스를 그린 확인용 그림을 <폴더>/_review 에 저장")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    if not os.path.isdir(a.folder):
        print("폴더가 없습니다 :", a.folder)
        return 1
    if not os.path.exists(a.model):
        print("모델이 없습니다 :", a.model)
        return 1

    jpgs = sorted(glob.glob(os.path.join(a.folder, "*.jpg")) +
                  glob.glob(os.path.join(a.folder, "*.JPG")))
    if not jpgs:
        print("사진(.jpg)이 없습니다 :", a.folder)
        return 1

    m = Model(a.model, a.threads)
    print("=" * 70)
    print(" 자동 라벨링")
    print("=" * 70)
    print("  폴더     :", os.path.abspath(a.folder))
    print("  사진     : %d장" % len(jpgs))
    print("  모델     : %s  (입력 %d)" % (m.name, m.size))
    print("  문턱값   : %.2f  (낮게 잡아 사람이 지우기 쉽게)" % a.conf)
    print("  이미 있는 .txt : %s" % ("덮어씀" if a.force else "건드리지 않음"))
    print()

    if a.review:
        os.makedirs(os.path.join(a.folder, "_review"), exist_ok=True)

    cnt = {c: 0 for c in NAMES}
    n_new = n_skip = 0
    empty = []          # 박스가 하나도 없는 사진 = 사람이 직접 그려야 할 것

    for i, p in enumerate(jpgs, 1):
        txt = os.path.splitext(p)[0] + ".txt"
        if os.path.exists(txt) and not a.force:
            n_skip += 1
            continue
        img = imread_any(p)
        if img is None:
            print("  못 읽음 :", os.path.basename(p))
            continue
        H, W = img.shape[:2]
        dets = m.detect(img, a.conf)

        lines = []
        for (x1, y1, x2, y2, sc, c) in dets:
            # YOLO 형식 : 클래스 중심x 중심y 너비 높이 (전부 0~1 비율)
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            bw, bh = min(bw, 1.0), min(bh, 1.0)
            lines.append("%d %.6f %.6f %.6f %.6f" % (c, cx, cy, bw, bh))
            cnt[c] += 1

        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_new += 1
        if not lines:
            empty.append(os.path.basename(p))

        if a.review:
            v = img.copy()
            for (x1, y1, x2, y2, sc, c) in dets:
                col = COLORS.get(c, (255, 255, 255))
                cv2.rectangle(v, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(v, "%s %d%%" % (NAMES.get(c, c), int(sc * 100)),
                            (int(x1), max(12, int(y1) - 6)),
                            cv2.FONT_HERSHEY_PLAIN, 1, col, 2)
            if not dets:
                cv2.putText(v, "NO BOX - draw by hand", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(a.folder, "_review", os.path.basename(p)), v)

        if i % 20 == 0:
            print("   %d/%d" % (i, len(jpgs)))

    # classes.txt — LabelImg 가 이 순서로 번호를 매긴다. 반드시 이 4줄이어야 한다.
    with open(os.path.join(a.folder, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("baby\nadult\nknife\noutlet\n")

    # ── 결과 ──
    print()
    print("=" * 70)
    print(" 결과")
    print("=" * 70)
    print("  라벨 만든 사진 : %d장" % n_new)
    if n_skip:
        print("  건너뛴 사진    : %d장 (이미 .txt 가 있음)" % n_skip)
    print()
    for c, nm in NAMES.items():
        print("    %d %-8s %4d개" % (c, nm, cnt[c]))
    print()
    print("  classes.txt 를 만들었습니다 (baby / adult / knife / outlet)")

    if empty:
        print()
        print("  ** 박스가 하나도 없는 사진 %d장 — 직접 그려야 합니다 **" % len(empty))
        print("     모델이 못 찾은 것들이고, 보강하려는 데이터가 바로 이것입니다.")
        for n in empty[:20]:
            print("       ", n)
        if len(empty) > 20:
            print("        ... 외 %d장" % (len(empty) - 20))
        with open(os.path.join(a.folder, "_손으로_그릴_사진.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(empty) + "\n")
        print("     목록 : %s" % os.path.join(a.folder, "_손으로_그릴_사진.txt"))

    print()
    print("=" * 70)
    print(" 다음에 할 것")
    print("=" * 70)
    print("  1) 이 폴더를 PC 로 가져옵니다")
    if a.review:
        print("  2) _review 폴더의 그림을 넘겨 보며 어디가 틀렸는지 먼저 훑습니다")
        print("     (사진 보기로 방향키만 누르면 되니 LabelImg 보다 빠릅니다)")
    else:
        print("  2) --review 를 붙여 다시 돌리면 확인용 그림이 생깁니다")
    print("  3) LabelImg 를 열어 고칩니다")
    print("       - 저장 형식을 **YOLO** 로 (왼쪽 아래 버튼)")
    print("       - Open Dir 로 이 폴더를 엽니다")
    print("       - 박스가 틀렸으면 고치고, 없으면 그립니다")
    print("  4) 다 되면 split_dataset.py 로 train/test 를 나눕니다")
    print()
    print("  ※ classes.txt 를 지우거나 순서를 바꾸지 마세요.")
    print("     LabelImg 가 그 줄 순서로 클래스 번호를 정합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
