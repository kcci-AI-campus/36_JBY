# -*- coding: utf-8 -*-
# EX_13 개선 실험판 (원본·트레이싱판과 별개 파일)
#
# 아래 4개의 스위치를 바꿔 가며 돌리면, 무엇이 얼마나 효과가 있는지 표로 비교할 수 있다.
# 스위치 조합이 파일 이름에 들어가므로 여러 번 돌려도 JSON이 안 덮어써진다.
#
#   실행:  python3 EX_13_opt.py
#   종료:  창을 클릭하고 q
#
# ─────────────────────────────────────────────────────────────
#  설정 (여기만 고치면 된다)
# ─────────────────────────────────────────────────────────────
WAIT_MS       = 1      # cv2.waitKey 대기(ms).  원본 10 → 1단계에서 1로
CAMERA_THREAD = True   # 카메라를 별도 스레드에서 미리 읽기 (2단계)
DETECT_EVERY  = 1      # 손 검출 주기. 1=매 프레임(원본), 2=두 프레임에 한 번 (3단계)
INTERP_THREADS = None  # 추론 스레드 수. None=기본값, 4=네 코어 모두 사용
FOURCC         = 'MJPG'  # 웹캠 전송 형식. 무압축 YUYV보다 장수를 많이 받는다
CAM_FPS        = 30      # 카메라에 요청할 프레임률. 0이면 요청 안 함
# ─────────────────────────────────────────────────────────────

import sys, os, threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_util import Tracer

from ai_edge_litert.interpreter import Interpreter
import numpy as np
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
hd = HandDetector(maxHands=1)

tag = 'w%d_cam%d_det%d_th%s' % (WAIT_MS, int(CAMERA_THREAD), DETECT_EVERY,
                               INTERP_THREADS if INTERP_THREADS else 'def')
print('=' * 58)
print(' 설정: waitKey=%dms | 카메라스레드=%s | 손검출주기=%d | 추론스레드=%s'
      % (WAIT_MS, CAMERA_THREAD, DETECT_EVERY, INTERP_THREADS))
print('=' * 58)

tr = Tracer('trace_opt_%s.json' % tag)

# TFLite 모델 로딩
modelPath = 'RPS_MobileNetV2_Augmentation_PTQ_INT8.tflite'
if INTERP_THREADS:
    try:
        interpreter = Interpreter(model_path=modelPath, num_threads=INTERP_THREADS)
    except TypeError:
        print('[경고] 이 버전은 num_threads를 지원하지 않습니다. 기본값으로 진행합니다.')
        interpreter = Interpreter(model_path=modelPath)
else:
    interpreter = Interpreter(model_path=modelPath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']
input_scale, input_zero = interpreter.get_input_details()[0]['quantization']
output_scale, output_zero = interpreter.get_output_details()[0]['quantization']
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('model input shape:', (height, width))

ansToText = {0:'scissors', 1:'rock', 2:'paper'}
colorList = [(255,0,0),(0,255,0),(0,0,255)]
IMG_SIZE = 224
offset = 30


class CameraThread:
    """카메라를 계속 읽어 최신 프레임만 들고 있는 스레드.
    메인 루프는 기다리지 않고 바로 최신 프레임을 가져간다."""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self.seq = 0          # 카메라가 실제로 내놓은 새 프레임 번호
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


def make_square_img(img):
    ho, wo = img.shape[0], img.shape[1]
    aspectRatio = ho/wo
    wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    if aspectRatio > 1:
        k = IMG_SIZE/ho
        wk = int(wo*k)
        img = cv2.resize( img, (wk, IMG_SIZE))
        img_h, img_w = img.shape[0], img.shape[1]
        d = (IMG_SIZE - img_w) // 2
        wbg [:img_h, d:img_w+d] = img
    else:
        k = IMG_SIZE/wo
        hk = int(ho*k)
        img = cv2.resize( img, (IMG_SIZE, hk))
        img_h, img_w = img.shape[0], img.shape[1]
        d = (IMG_SIZE - img_h) // 2
        wbg [d:img_h+d, :img_w ] = img
    return wbg


last_bbox = None   # 3단계: 검출을 건너뛴 프레임에서 재사용할 박스

def processImage(frame, frame_idx):
    global last_bbox

    # 3단계: DETECT_EVERY 프레임마다 한 번만 검출한다
    if frame_idx % DETECT_EVERY == 0:
        with tr.span('handDetect'):
            hands, _ = hd.findHands(frame, draw=False)
        last_bbox = hands[0]['bbox'] if hands else None
    else:
        tr.instant('skip-detect')

    if last_bbox is None:
        tr.instant('no-hand')
        return

    with tr.span('preproc'):
        x, y, w, h = last_bbox

        if x<offset or y<offset or x+w+offset>320 or y+h>240:
            tr.instant('out-of-range')
            return

        x1, y1 = x-offset,  y-offset
        x2, y2 = x+w+offset, y+h

        img = frame[y1:y2, x1:x2]
        img = make_square_img(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        img = (img / input_scale) + input_zero

    with tr.span('inference'):
        interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
        interpreter.invoke()
        output_tensor = interpreter.get_tensor(output_details[0]['index'])[0]

    with tr.span('postproc'):
        predictions = (output_tensor.astype(np.float32) - output_zero) * output_scale
        ans = np.argmax(predictions)
        text = ansToText[ans]

    with tr.span('drawBox'):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[ans], 2)
        cv2.putText(frame,text,(x1,y1-7),cv2.FONT_HERSHEY_PLAIN,2,colorList[ans],2)


# 카메라 설정
cap = cv2.VideoCapture(0)
if FOURCC != 'NONE':
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
if CAM_FPS:
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

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

source = CameraThread(cap) if CAMERA_THREAD else DirectCamera(cap)

# 윈도우 설정
cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('cam', 320+40, 240+60)

# 첫 프레임이 들어올 때까지 기다린다 (측정 시작 전에)
if CAMERA_THREAD:
    for _ in range(500):            # 최대 5초
        ok, f, _ = source.read()
        if ok and f is not None:
            break
        time.sleep(0.01)
    else:
        print('[오류] 카메라에서 프레임을 받지 못했습니다.')
        source.release(); cap.release(); sys.exit(1)
    print('[camera] 첫 프레임 도착. 측정을 시작합니다.')

frame_idx = 0
dup_count = 0          # 카메라가 새 프레임을 못 준 채로 처리한 횟수
last_seq = -1
t_start_all = time.time()
startTime = time.time()
try:
    while True:
        with tr.span('frame'):
            with tr.span('camera'):
                ret, frame, seq = source.read()
            if not ret or frame is None:
                break
            if seq == last_seq:
                dup_count += 1
                tr.instant('duplicate-frame')
            last_seq = seq

            processImage(frame, frame_idx)
            frame_idx += 1

            curTime = time.time()
            fps = 1/(curTime - startTime)
            startTime = curTime
            with tr.span('overlay'):
                cv2.putText(frame,f'FPS: {fps:.1f}',(20, 50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),2)

            with tr.span('imshow'):
                cv2.imshow('cam',frame)

            with tr.span('waitKey'):
                key = cv2.waitKey(WAIT_MS)
        if key == ord('q'): break
except KeyboardInterrupt:
    print('\n[trace] Ctrl+C 감지')

elapsed = time.time() - t_start_all
tr.save()
if frame_idx:
    new_frames = frame_idx - dup_count
    print('-' * 58)
    print(' 처리한 프레임 : %d  (초당 %.1f)' % (frame_idx, frame_idx/elapsed))
    print(' 그중 중복     : %d  (%.1f%%)  <- 카메라가 새 그림을 못 준 프레임'
          % (dup_count, dup_count/frame_idx*100))
    print(' 실제 새 프레임: %d  (초당 %.1f)  <- 카메라가 낼 수 있는 한계'
          % (new_frames, new_frames/elapsed))
    if dup_count/frame_idx > 0.05:
        print(' [주의] 중복이 5%%를 넘습니다. 파이프라인이 카메라보다 빨라졌다는 뜻이고,')
        print('        FPS 숫자가 실제 처리량을 부풀리고 있습니다.')
    print('=' * 58)
source.release()
cap.release()
cv2.destroyAllWindows()
