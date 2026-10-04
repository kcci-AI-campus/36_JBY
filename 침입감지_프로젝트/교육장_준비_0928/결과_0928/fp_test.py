# 칼·인형 없는 장면 60장: 모델마다 knife/outlet/baby 로 잘못 잡힌 프레임 수 (문턱 0.25/0.40/0.50)
import cv2, numpy as np, time
from ai_edge_litert.interpreter import Interpreter
cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(15): cap.read()
frames = []
for _ in range(60):
    ok, f = cap.read(); frames.append(f); time.sleep(0.05)
cap.release()
def infer(it, i, o, f):
    nchw = i["shape"][1] == 3; S = int(i["shape"][2] if nchw else i["shape"][1]); H, W = f.shape[:2]
    r = min(S / W, S / H); nw, nh = int(W * r), int(H * r); px, py = (S - nw) // 2, (S - nh) // 2
    x = np.full((S, S, 3), 114, np.uint8); x[py:py + nh, px:px + nw] = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (nw, nh))
    x = (x.astype(np.float32) / 255.0)[None]
    if nchw: x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    it.set_tensor(i["index"], x); it.invoke(); raw = it.get_tensor(o["index"])[0]
    if raw.shape[0] < raw.shape[1]: raw = raw.T
    return raw, S, r, px, py
MODELS = [("YOLO11n FP32 320", "yolo11n_fp32.tflite", (0, 0, 255)), ("YOLO11n INT8 320", "yolo11n_int8.tflite", (0, 140, 255)),
          ("YOLOv8n INT8 320", "/tmp/yolo8vn_int8.tflite", (0, 200, 0)), ("YOLO11n FP32 640", "../v2e120_fp32_640.tflite", (255, 0, 0))]
print("모델                잘못 잡힌 프레임 수 / 60장   knife@0.25  knife@0.40  knife@0.50 | outlet@0.40 baby@0.40 | knife 최고점")
vis = frames[-1].copy()
for label, m, col in MODELS:
    it = Interpreter(model_path=m); it.allocate_tensors(); i = it.get_input_details()[0]; o = it.get_output_details()[0]
    k = {0.25: 0, 0.40: 0, 0.50: 0}; ot = 0; bb = 0; kmax = 0.0
    for n, f in enumerate(frames):
        raw, S, r, px, py = infer(it, i, o, f)
        ks = raw[:, 6]; km = float(ks.max()); kmax = max(kmax, km)
        for t in k: k[t] += km >= t
        ot += raw[:, 7].max() >= 0.40; bb += raw[:, 4].max() >= 0.40
        if n == len(frames) - 1 and km >= 0.25:
            j = int(ks.argmax()); cx, cy, w, h = raw[j, :4]
            if abs(raw[j, :4]).max() <= 2.0: cx, cy, w, h = cx * S, cy * S, w * S, h * S
            x1, y1, x2, y2 = int((cx - w / 2 - px) / r), int((cy - h / 2 - py) / r), int((cx + w / 2 - px) / r), int((cy + h / 2 - py) / r)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2); cv2.putText(vis, "%s %.2f" % (label.split()[0][-3:] + label.split()[-1], km), (x1, max(12, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
    print("%-18s %32s %10d %11d %11d | %11d %9d | %.2f" % (label, "", k[0.25], k[0.40], k[0.50], ot, bb, kmax))
cv2.imwrite("/tmp/fp_vis.jpg", vis)
