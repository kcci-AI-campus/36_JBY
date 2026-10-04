# -*- coding: utf-8 -*-
# EX_03 개선 실험판 (원본·트레이싱판과 별개 파일)
#
# 04 실습(EX_13_opt.py)에서 효과를 확인한 개선을 검출 파이프라인에 그대로 적용한다.
# 아래 스위치를 바꿔 가며 돌리면 무엇이 얼마나 효과가 있는지 표로 비교할 수 있다.
#
#   실행:  python3 EX_03_opt.py my_best_int8.tflite
#   종료:  창을 클릭하고 q
#
# ─────────────────────────────────────────────────────────────
#  설정 (여기만 고치면 된다)
# ─────────────────────────────────────────────────────────────
WAIT_MS        = 1     # cv2.waitKey 대기(ms). 원본 10 -> 1
CAMERA_THREAD  = True  # 카메라를 별도 스레드에서 미리 읽기
INFER_EVERY    = 1     # 추론 주기. 1=매 프레임, 2=두 프레임에 한 번(이전 박스 재사용)
INTERP_THREADS = 4     # 추론 스레드 수. None=기본값, 4=네 코어 모두 사용
FOURCC         = 'MJPG'  # 웹캠 전송 형식. 무압축 YUYV보다 장수를 많이 받는다
CAM_FPS        = 30      # 카메라에 요청할 프레임률. 0이면 요청 안 함
#                        <- 추론이 전체의 84%이므로 여기가 가장 중요하다
# ─────────────────────────────────────────────────────────────

# 모듈 로딩
import sys, os, threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_util import Tracer

# LiteRT 인터프리터를 가져온다.
# 같은 물건인데 패키지 이름이 바뀌었다.
#   옛 이름 : tflite_runtime.interpreter   (구글이 TensorFlow Lite로 부르던 시절)
#   새 이름 : ai_edge_litert.interpreter   (LiteRT로 이름이 바뀐 뒤)
# 강사님 공지(2026-09-17)로 05 예제는 새 이름을 쓰기로 했다.
# 보드에 둘 중 무엇이 깔려 있든 돌아가도록 새 이름을 먼저 시도한다.
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2

# LiteRT 모델 선택 (인자로 주면 그것을, 없으면 기본값)
modelPath = sys.argv[1] if len(sys.argv) > 1 else "best.tflite"
#modelPath = "best_int8.tflite"
#modelPath = "best_w8a32.tflite"
print('model path:', modelPath)

tag = '%s_w%d_cam%d_inf%d_th%s' % (
    os.path.splitext(os.path.basename(modelPath))[0],
    WAIT_MS, int(CAMERA_THREAD), INFER_EVERY,
    INTERP_THREADS if INTERP_THREADS else 'def')
print('=' * 58)
print(' 설정: waitKey=%dms | 카메라스레드=%s | 추론주기=%d | 추론스레드=%s'
      % (WAIT_MS, CAMERA_THREAD, INFER_EVERY, INTERP_THREADS))
print('=' * 58)

tr = Tracer('trace_opt_%s.json' % tag)

# LiteRT 모델 로딩
if INTERP_THREADS:
    try:
        interpreter = tflite.Interpreter(model_path=modelPath, num_threads=INTERP_THREADS)
    except TypeError:
        print('[경고] 이 버전은 num_threads를 지원하지 않습니다. 기본값으로 진행합니다.')
        interpreter = tflite.Interpreter(model_path=modelPath)
else:
    interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors() # tensor 할당

# 모델 정보 얻기
input_details = interpreter.get_input_details()  # input tensor 정보 얻기
output_details = interpreter.get_output_details() # output tensor 정보 얻기
print(input_details)
print(output_details)
input_index = input_details[0]['index']
output_index = output_details[0]['index']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
height = input_details[0]['shape'][2]
width = input_details[0]['shape'][3]
print('model input shape:', (height, width))

# BB 텍스트 및 색상 정의
ansToText = {0:'scissors', 1:'rock', 2:'paper'}
colorList = [(255,0,0),(0,255,0),(0,0,255)]

# 모델 입력 크기
IMG_SIZE = 320

# Threshold 설정
CONF_TH = 0.4
IOU_TH  = 0.45

class CameraThread:
    """카메라를 계속 읽어 최신 프레임만 들고 있는 스레드."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self.seq = 0
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop_flag.is_set():
            ok, f = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, f
                self.seq += 1
            if not ok:
                break

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.seq
            return self.ok, self.frame.copy(), self.seq

    def release(self):
        self.stop_flag.set()
        self.thread.join(timeout=1.0)


class DirectCamera:
    """카메라 스레드를 안 쓸 때. read()의 반환 형식을 맞춰 준다."""
    def __init__(self, cap):
        self.cap = cap
        self.seq = 0
    def read(self):
        ok, f = self.cap.read()
        self.seq += 1
        return ok, f, self.seq
    def release(self):
        pass


def letterbox(img, new_shape=(320,320), color=(114,114,114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_y, pad_y,
        pad_x, pad_x,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, r, pad_x, pad_y

last_draw = None   # 추론을 건너뛴 프레임에서 재사용할 결과

def processImage(frame, frame_idx):
    global last_draw

    if frame_idx % INFER_EVERY != 0:
        tr.instant('skip-infer')
        if last_draw is not None:
            with tr.span('drawBox'):
                drawResult(frame, last_draw)
        return

    with tr.span('preproc'):
        # BGR을 RGB로 변경
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # letterbox 적용
        img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))

        # 0 ~ 1 사이 값으로 변경
        img = img_lb.astype(np.float32) / 255.0

        # 모델의 입력 형태로 수정: (1,3,320,320)
        #   최상위 차원 증가: (320,320,3) -> (1,320,320,3)
        img = np.expand_dims(img, axis=0)
        #   축 위치 변경: (1,320,320,3) -> (1,3,320,320)
        img = np.transpose(img, (0, 3, 1, 2))

    # 모델에 입력하여 결과 얻기
    with tr.span('inference'):
        #   input tensor 설정
        interpreter.set_tensor(input_index, img)
        #   모델 실행
        interpreter.invoke()
        #   output tensor 얻기: (1,7,2100) -> (7,2100) -> (2100,7)
        raw = interpreter.get_tensor(output_index)[0].transpose()

    with tr.span('filter'):
        # raw에서 각 Object 별로 confidence score의 최대값과 해당 클래스의 id를 추출
        class_scores = raw[:, 4:]                      # confidence score: (2100, 3)
        confidences = np.max(class_scores, axis=1)     # confidence score의 최대값: (2100,)
        class_ids = np.argmax(class_scores, axis=1)    # 해당 클래스의 id: (2100,)

        # confidence score가 설정한 임계값(CONF_TH)보다 높은 Object만 필터링
        keep_mask = confidences > CONF_TH
        filtered_raw = raw[keep_mask] # (N,7)
        scores = confidences[keep_mask] # (N,)
        classes = class_ids[keep_mask]  # (N,)

        # 중심점 좌표 [cx, cy, w, h]를 추출하고 좌상단 좌표 [x, y, w, h]로 변환
        cx, cy, w, h = filtered_raw[:, 0], filtered_raw[:, 1], filtered_raw[:, 2], filtered_raw[:, 3]
        x = cx - (w / 2)
        y = cy - (h / 2)
        boxes = np.stack([x, y, w, h], axis=-1) # (N,4)

    tr.counter('candidates', {'after_conf_filter': int(len(scores))})

    with tr.span('nms'):
        # Batched NMS 처리
        keep = cv2.dnn.NMSBoxesBatched(
                boxes,
                scores,
                classes,
                score_threshold=CONF_TH,
                nms_threshold=IOU_TH
        )

        # 최종 BB만 그리기
        draw_boxes = [(boxes[i], scores[i], classes[i]) for i in keep]

    tr.counter('detections', {'after_nms': int(len(draw_boxes))})

    last_draw = (draw_boxes, r, pad_x, pad_y)

    with tr.span('drawBox'):
        drawResult(frame, last_draw)


def drawResult(frame, packed):
    draw_boxes, r, pad_x, pad_y = packed
    if True:
        for (x,y,w,h), sc, cid in draw_boxes:
            x *= IMG_SIZE; y *= IMG_SIZE
            w *= IMG_SIZE; h *= IMG_SIZE

            x1 = (x - pad_x) / r
            y1 = (y - pad_y) / r
            x2 = (x + w - pad_x) / r
            y2 = (y + h - pad_y) / r

            x1 = int(np.clip(x1,0,frame.shape[1]))
            y1 = int(np.clip(y1,0,frame.shape[0]))
            x2 = int(np.clip(x2,0,frame.shape[1]))
            y2 = int(np.clip(y2,0,frame.shape[0]))

            # BB 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)

            # 판정 결과 표시
            cv2.putText(frame, f'{ansToText[cid]} {int(sc*100)}%', (x1,y1-7), cv2.FONT_HERSHEY_PLAIN, 1, colorList[cid], 2)

# 카메라 설정
cap = cv2.VideoCapture(0) # 0번 카메라 열기
if FOURCC != 'NONE':
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
if CAM_FPS:
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

source = CameraThread(cap) if CAMERA_THREAD else DirectCamera(cap)

# 윈도우 설정
cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 320+40, 240+60)

# 첫 프레임이 들어올 때까지 기다린다 (측정 시작 전에)
if CAMERA_THREAD:
    for _ in range(500):
        ok, f, _ = source.read()
        if ok and f is not None:
            break
        time.sleep(0.01)
    else:
        print('[오류] 카메라에서 프레임을 받지 못했습니다.')
        source.release(); cap.release(); sys.exit(1)
    print('[camera] 첫 프레임 도착. 측정을 시작합니다.')

frame_idx = 0
dup_count = 0
last_seq = -1
t_start_all = time.time()
startTime = time.time()
try:
    while True:
        with tr.span('frame'):
            with tr.span('camera'):
                ret, frame, seq = source.read()
            if not ret or frame is None: break
            if seq == last_seq:
                dup_count += 1
                tr.instant('duplicate-frame')
            last_seq = seq

            # 이미지 처리
            processImage(frame, frame_idx)
            frame_idx += 1

            # FPS 표시
            curTime = time.time()
            fps = 1/(curTime - startTime)
            startTime = curTime
            with tr.span('overlay'):
                cv2.putText(frame,f'FPS: {fps:.1f}',(20, 50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)

            # 이미지 출력
            with tr.span('imshow'):
                cv2.imshow('cam',frame)

            # 10ms 동안 키 입력 대기
            with tr.span('waitKey'):
                key = cv2.waitKey(10)
        if  key == ord('q'): break
except KeyboardInterrupt:
    print('\n[trace] Ctrl+C 감지')

elapsed = time.time() - t_start_all
tr.save()
if frame_idx:
    new_frames = frame_idx - dup_count
    print('-' * 58)
    print(' 처리한 프레임 : %d  (초당 %.1f)' % (frame_idx, frame_idx/elapsed))
    print(' 그중 중복     : %d  (%.1f%%)' % (dup_count, dup_count/frame_idx*100))
    print(' 실제 새 프레임: %d  (초당 %.1f)' % (new_frames, new_frames/elapsed))
    if dup_count/frame_idx > 0.05:
        print(' [주의] 중복이 5%%를 넘습니다. 파이프라인이 카메라보다 빨라졌다는 뜻입니다.')
    print('=' * 58)
source.release()
cap.release() # 카메라 닫기
cv2.destroyAllWindows() # 모든 창 닫기
