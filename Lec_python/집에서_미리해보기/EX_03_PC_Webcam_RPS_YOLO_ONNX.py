# 가위바위보 YOLO11n 실시간 인식 - 노트북 웹캠 + ONNX 버전
#
# 원본: 02.examples(Board).zip → examples/05_Object_Detection_Based_On-Device_AI/EX_03_Board_RPS_PreTrained_YOLO.py
# 원본은 라즈베리파이에서 tflite 모델을 돌리는 코드이고, 이 파일은 같은 흐름을 윈도우 노트북에서 돌리도록 바꾼 것.
# 원본과 달라진 곳에는 [변경] 표시를 달아 둠.
#
# 실행 방법 (이 폴더에서):
#   .venv\Scripts\python.exe EX_03_PC_Webcam_RPS_YOLO_ONNX.py            → 웹캠 실시간 (q 키로 종료)
#   .venv\Scripts\python.exe EX_03_PC_Webcam_RPS_YOLO_ONNX.py --cam 1    → 다른 카메라 번호 사용
#   .venv\Scripts\python.exe EX_03_PC_Webcam_RPS_YOLO_ONNX.py --image 사진.jpg → 사진 한 장 판정 후 _result.jpg 저장

# 모듈 로딩
import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort  # [변경] tflite_runtime 대신 ONNX Runtime 사용

# ONNX 모델 경로 [변경] tflite 대신 교육 폴더의 rps_yolo11n.onnx 사용
HERE = os.path.dirname(os.path.abspath(__file__))
modelPath = glob.glob(os.path.join(HERE, '..', '2', 'Deep_Learning*', '02.DL(CNN)_Files', 'save', 'rps_yolo11n.onnx'))[0]
print('model path:', os.path.normpath(modelPath))

# ONNX 모델 로딩 [변경] Interpreter + allocate_tensors 대신 InferenceSession 한 줄
session = ort.InferenceSession(modelPath, providers=['CPUExecutionProvider'])

# 모델 정보 얻기
input_info = session.get_inputs()[0]    # 이름 'images', 형태 [1, 3, 320, 320]
output_info = session.get_outputs()[0]  # 이름 'output0', 형태 [1, 7, 2100]
print('input :', input_info.name, input_info.shape)
print('output:', output_info.name, output_info.shape)
height = input_info.shape[2]
width = input_info.shape[3]
print('model input shape:', (height, width))

# BB 텍스트 및 색상 정의
ansToText = {0: 'scissors', 1: 'rock', 2: 'paper'}
colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 모델 입력 크기
IMG_SIZE = 320

# Threshold 설정
CONF_TH = 0.4
IOU_TH = 0.45


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    # [변경] 남는 칸이 홀수일 때 1픽셀 모자라지 않도록 오른쪽/아래에 나머지를 더함
    padded = cv2.copyMakeBorder(
        resized,
        pad_y, pad_h - pad_y,
        pad_x, pad_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, r, pad_x, pad_y


def processImage(frame):

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

    # 모델에 입력하여 결과 얻기 [변경] set_tensor → invoke → get_tensor 3단계가 run 한 줄로
    #   output: (1,7,2100) -> (7,2100) -> (2100,7)
    raw = session.run([output_info.name], {input_info.name: img})[0][0].transpose()

    # raw에서 각 Object 별로 confidence score의 최대값과 해당 클래스의 id를 추출
    class_scores = raw[:, 4:]                      # confidence score: (2100, 3)
    confidences = np.max(class_scores, axis=1)     # confidence score의 최대값: (2100,)
    class_ids = np.argmax(class_scores, axis=1)    # 해당 클래스의 id: (2100,)

    # confidence score가 설정한 임계값(CONF_TH)보다 높은 Object만 필터링
    keep_mask = confidences > CONF_TH
    filtered_raw = raw[keep_mask]  # (N,7)
    scores = confidences[keep_mask]  # (N,)
    classes = class_ids[keep_mask]  # (N,)

    # 중심점 좌표 [cx, cy, w, h]를 추출하고 좌상단 좌표 [x, y, w, h]로 변환
    cx, cy, w, h = filtered_raw[:, 0], filtered_raw[:, 1], filtered_raw[:, 2], filtered_raw[:, 3]
    x = cx - (w / 2)
    y = cy - (h / 2)
    boxes = np.stack([x, y, w, h], axis=-1)  # (N,4)

    # Batched NMS 처리
    keep = cv2.dnn.NMSBoxesBatched(
            boxes,
            scores,
            classes,
            score_threshold=CONF_TH,
            nms_threshold=IOU_TH
    )

    # 최종 BB만 그리기
    draw_boxes = [(boxes[i], scores[i], classes[i]) for i in np.array(keep).flatten()]

    for (x, y, w, h), sc, cid in draw_boxes:
        # [변경] tflite 모델은 좌표가 0~1 비율이라 IMG_SIZE를 곱했지만,
        #        ONNX 모델은 좌표가 이미 0~320 픽셀 단위라서 곱하지 않음

        x1 = (x - pad_x) / r
        y1 = (y - pad_y) / r
        x2 = (x + w - pad_x) / r
        y2 = (y + h - pad_y) / r

        x1 = int(np.clip(x1, 0, frame.shape[1]))
        y1 = int(np.clip(y1, 0, frame.shape[0]))
        x2 = int(np.clip(x2, 0, frame.shape[1]))
        y2 = int(np.clip(y2, 0, frame.shape[0]))

        # BB 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[cid], 2)

        # 판정 결과 표시
        cv2.putText(frame, f'{ansToText[cid]} {int(sc*100)}%', (x1, max(y1-7, 15)), cv2.FONT_HERSHEY_PLAIN, 1.5, colorList[cid], 2)

    return draw_boxes


def runImage(path):
    # [추가] 사진 한 장으로 판정해 보기 (웹캠 없이 동작 확인용)
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 한글 경로도 읽히게
    results = processImage(frame)
    for (x, y, w, h), sc, cid in results:
        print(f'{ansToText[cid]} {sc:.2f}')
    if not results:
        print('검출 없음')
    out = os.path.splitext(path)[0] + '_result.jpg'
    cv2.imencode('.jpg', frame)[1].tofile(out)
    print('saved:', out)


def runWebcam(camIndex):
    # 카메라 설정 [변경] 윈도우에서 빨리 열리도록 DirectShow(CAP_DSHOW) 사용
    cap = cv2.VideoCapture(camIndex, cv2.CAP_DSHOW)
    # [변경] 노트북 웹캠은 보드보다 빨라서 320x240 대신 640x480으로 받음
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f'{camIndex}번 카메라를 열 수 없습니다. --cam 1 처럼 다른 번호를 넣어 보세요.')
        return

    # 윈도우 설정
    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cam', 640+40, 480+60)

    startTime = time.time()
    while cap.isOpened():
        ret, frame = cap.read()  # 사진 찍기 -> (480,640,3)
        if not ret:
            # [추가] 카메라에서 화면을 못 받으면 조용히 끝나지 않고 이유를 알려 줌
            print(f'{camIndex}번 카메라에서 화면을 받지 못했습니다. 다른 앱(Zoom, Teams 등)이 카메라를 쓰고 있지 않은지 확인하거나 --cam 1 을 넣어 보세요.')
            break

        # [변경] 거울 모드: 화면이 거울처럼 보이게 좌우 반전 (손 모양 판정에는 영향 없음)
        frame = cv2.flip(frame, 1)

        # 이미지 처리
        processImage(frame)

        # FPS 표시
        curTime = time.time()
        fps = 1/(curTime - startTime)
        startTime = curTime
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # 이미지 출력
        cv2.imshow('cam', frame)

        # 10ms 동안 키 입력 대기 (q 또는 창 닫기로 종료)
        key = cv2.waitKey(10)
        if key == ord('q') or cv2.getWindowProperty('cam', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()  # 카메라 닫기
    cv2.destroyAllWindows()  # 모든 창 닫기


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0, help='카메라 번호 (기본 0)')
    parser.add_argument('--image', help='웹캠 대신 사진 한 장으로 판정')
    args = parser.parse_args()

    if args.image:
        runImage(args.image)
    else:
        runWebcam(args.cam)
