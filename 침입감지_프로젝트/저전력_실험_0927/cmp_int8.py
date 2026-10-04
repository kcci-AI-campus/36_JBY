# FP32 모델과 INT8 모델이 같은 프레임에서 같은 것을 찾는지 비교한다 (정확도 대리 지표).
#   입력: 9/23 경보 클립 32장 + far/guard/static 프레임. 출력: 클래스별 검출 일치율, 신뢰도 차이, 박스 IoU.
#   진짜 정확도(mAP)는 정답 라벨이 있는 시험 사진으로 재야 한다. 여기서는 "FP32 가 찾은 것을 INT8 도 찾나" 만 본다.
import sys, cv2, numpy as np
from ai_edge_litert.interpreter import Interpreter
NAMES = {0: "baby", 1: "adult", 2: "knife", 3: "outlet"}
CONF = 0.40


class Yolo:
    def __init__(self, path):
        self.it = Interpreter(model_path=path, num_threads=4); self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]; self.out = self.it.get_output_details()[0]
        self.S = int(self.inp["shape"][2]); self.int_in = self.inp["dtype"] in (np.int8, np.uint8)
        self.q = self.inp.get("quantization", (0.0, 0))

    def __call__(self, img):
        H, W = img.shape[:2]; S = self.S
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); r = min(S / W, S / H); nw, nh = int(W * r), int(H * r)
        x = np.full((S, S, 3), 114, np.uint8); px, py = (S - nw) // 2, (S - nh) // 2
        x[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh))
        x = x.astype(np.float32) / 255.0
        if self.int_in:
            s, z = self.q; info = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(x / (s if s else 1.0) + z), info.min, info.max).astype(self.inp["dtype"])
        x = np.ascontiguousarray(np.transpose(x[None], (0, 3, 1, 2)))
        self.it.set_tensor(self.inp["index"], x); self.it.invoke()
        raw = self.it.get_tensor(self.out["index"])
        if self.out["dtype"] in (np.int8, np.uint8):
            s, z = self.out.get("quantization", (0.0, 0)); raw = (raw.astype(np.float32) - z) * s
        raw = raw[0].T
        conf = raw[:, 4:].max(1); cls = raw[:, 4:].argmax(1)
        b = raw[:, :4].copy()
        if (conf >= CONF).any() and np.abs(b[conf >= CONF]).max() <= 2.0: b *= S
        boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        m = conf > CONF
        if not m.any(): return []
        boxes, conf, cls = boxes[m], conf[m], cls[m]
        xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
        keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), CONF, 0.45)
        res = []
        for i in np.array(keep).flatten():
            x1, y1, x2, y2 = boxes[i]
            res.append((int(cls[i]), float(conf[i]), (x1 - px) / r, (y1 - py) / r, (x2 - px) / r, (y2 - py) / r))
        return res


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


frames = []
cap = cv2.VideoCapture("clip.mp4")
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(("clip", f))
for n in ("static_frame.jpg", "far60_frame.jpg", "guard60_frame.jpg", "babyonly_frame.jpg", "empty.jpg", "best.jpg"):
    img = cv2.imread(n)
    if img is not None: frames.append((n, img))
print("프레임", len(frames), "장")
A = Yolo(sys.argv[1]); B = Yolo(sys.argv[2])
stat = {c: {"a": 0, "b": 0, "both": 0, "iou": [], "dconf": []} for c in NAMES}
for name, img in frames:
    da, db = A(img), B(img)
    for c in NAMES:
        la = [d for d in da if d[0] == c]; lb = [d for d in db if d[0] == c]
        stat[c]["a"] += len(la); stat[c]["b"] += len(lb)
        for d in la:
            best = max(lb, key=lambda e: iou(d[2:], e[2:]), default=None)
            if best and iou(d[2:], best[2:]) >= 0.5:
                stat[c]["both"] += 1; stat[c]["iou"].append(iou(d[2:], best[2:])); stat[c]["dconf"].append(best[1] - d[1])
print("%-8s %8s %8s %10s %9s %10s" % ("클래스", "FP32검출", "INT8검출", "둘다(IoU≥.5)", "평균IoU", "신뢰도차이"))
for c, s in stat.items():
    print("%-8s %8d %8d %10d %9s %10s" % (NAMES[c], s["a"], s["b"], s["both"],
          ("%.2f" % np.mean(s["iou"])) if s["iou"] else "-", ("%+.3f" % np.mean(s["dconf"])) if s["dconf"] else "-"))
print("해석: '둘다' 가 'FP32검출' 에 가까우면 INT8 이 같은 물체를 찾는다. INT8검출 > FP32검출 이면 INT8 이 더 잡는다(오검출일 수도).")
