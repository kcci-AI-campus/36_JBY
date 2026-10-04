# 경계(guard) 상태 시험용 영상 두 개를 만든다 (모두 640x480, 20 FPS, 65초, 정지 장면)
#   far60.mp4   : 인형(왼쪽) + 칼(오른쪽 끝)  -> 거리 > 2 x 위험 반경  => 감시 상태에 머물러야 한다
#   guard60.mp4 : 인형(왼쪽) + 칼(가운데 오른쪽) -> 반경 < 거리 < 2 x 반경 => 경계 상태여야 한다
#   만든 뒤 YOLO11n 320 으로 한 장씩 검출해 실제 거리/반경을 찍는다.
import cv2, numpy as np
from ai_edge_litert.interpreter import Interpreter
f = cv2.imread("static_frame.jpg")
doll = cv2.resize(f[40:480, 70:330], None, fx=0.8, fy=0.8)          # 208 x 352
knife = f[130:478, 336:436]                                          # 100 x 348 (칼 + 손 일부)
H, W, FPS = 480, 640, 20


def compose(knife_x):
    c = np.full((H, W, 3), (120, 120, 120), np.uint8)
    dh, dw = doll.shape[:2]; kh, kw = knife.shape[:2]
    c[H - dh - 20:H - 20, 0:dw] = doll
    kx = min(knife_x, W - kw)
    c[H - kh - 40:H - 40, kx:kx + kw] = knife
    return c


NAMES = {0: "baby", 1: "adult", 2: "knife", 3: "outlet"}
it = Interpreter(model_path="yolo11n_fp32.tflite", num_threads=4); it.allocate_tensors()
inp = it.get_input_details()[0]; out = it.get_output_details()[0]; S = 320


def detect(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); r = min(S / W, S / H); nw, nh = int(W * r), int(H * r)
    x = np.full((S, S, 3), 114, np.uint8); px, py = (S - nw) // 2, (S - nh) // 2
    x[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh))
    x = np.transpose(x.astype(np.float32)[None] / 255.0, (0, 3, 1, 2))
    it.set_tensor(inp["index"], np.ascontiguousarray(x)); it.invoke()
    raw = it.get_tensor(out["index"])[0].T; conf = raw[:, 4:].max(1); cls = raw[:, 4:].argmax(1)
    best = {}
    for i in np.argsort(-conf):
        if conf[i] < 0.4: break
        c = int(cls[i])
        if c in best: continue
        cx, cy, w, h = raw[i, :4] * S
        best[c] = (conf[i], int((cx - w / 2 - px) / r), int((cy - h / 2 - py) / r), int((cx + w / 2 - px) / r), int((cy + h / 2 - py) / r))
    return best


for name, kx in (("far60", 560), ("guard60", 400)):
    img = compose(kx); d = detect(img)
    line = []
    for c, (sc, x1, y1, x2, y2) in d.items():
        line.append("%s %.2f (%d,%d,%d,%d)" % (NAMES[c], sc, x1, y1, x2, y2))
    gap = radius = None
    if 0 in d and 2 in d:
        b = d[0][1:]; k = d[2][1:]
        dx = max(k[0] - b[2], b[0] - k[2], 0); dy = max(k[1] - b[3], b[1] - k[3], 0)
        gap = (dx * dx + dy * dy) ** 0.5; radius = (k[2] - k[0]) * 1.5
    print("%-8s 검출: %s | 거리 %s / 반경 %s / 비율 %s" % (name, "; ".join(line), None if gap is None else round(gap), None if radius is None else round(radius), None if gap is None else round(gap / radius, 2)))
    vw = cv2.VideoWriter(name + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(65 * FPS): vw.write(img)
    vw.release(); cv2.imwrite(name + "_frame.jpg", img); print(name + ".mp4 ok")
