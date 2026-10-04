# 카메라 10장을 한 번만 찍어 같은 사진을 모델마다 넣고, 클래스별 최고 점수의 평균을 비교
import cv2, numpy as np, time
from ai_edge_litert.interpreter import Interpreter
NAMES = ["baby", "adult", "knife", "outlet"]
cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(15): cap.read()
frames = []
for _ in range(10):
    ok, f = cap.read(); frames.append(f); time.sleep(0.1)
cap.release(); cv2.imwrite("/tmp/cmp_frame.jpg", frames[-1])
def prep(img, S, nchw):
    h, w = img.shape[:2]; r = min(S / w, S / h); nw, nh = int(w * r), int(h * r)
    x = np.full((S, S, 3), 114, np.uint8); px, py = (S - nw) // 2, (S - nh) // 2
    x[py:py + nh, px:px + nw] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh))
    x = (x.astype(np.float32) / 255.0)[None]
    return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2))) if nchw else x
models = [("YOLO11n FP32", "yolo11n_fp32.tflite"), ("YOLO11n INT8", "yolo11n_int8.tflite"),
          ("YOLO26n INT8", "yolov26n_int8.tflite"), ("YOLOv8n INT8", "/tmp/yolo8vn_int8.tflite"), ("YOLOv10n INT8", "/tmp/yolo10n_int8.tflite")]
print("%-15s %s" % ("모델", "  ".join("%-7s" % n for n in NAMES)) + "   (10장 평균, 각 클래스 최고 점수)")
for label, m in models:
    it = Interpreter(model_path=m); it.allocate_tensors()
    i = it.get_input_details()[0]; o = it.get_output_details()[0]
    best = np.zeros((len(frames), 4))
    for k, f in enumerate(frames):
        it.set_tensor(i["index"], prep(f, 320, i["shape"][1] == 3)); it.invoke()
        raw = it.get_tensor(o["index"])[0]
        if raw.shape[0] < raw.shape[1]: raw = raw.T
        best[k] = raw[:, 4:8].max(0)
    print("%-15s %s" % (label, "  ".join("%-7.2f" % v for v in best.mean(0))))
