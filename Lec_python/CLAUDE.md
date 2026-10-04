# Lec_python

1개월차 과목: AI 프로그래밍 기초.

- `1_week_python/` : 파이썬 기초 (2026-09-01 ~ 09-08). VS Code 실습.
- `Deep_Learning(축약)_Lab_[64-301-01]/` : 딥러닝/CNN 실습 (2026-09-09 ~). Google Colab에서 진행.
  - `01.DL(CNN)_Examples/` 예제 노트북
  - `02.DL(CNN)_Files/` 데이터셋(RPS), 저장된 모델
  - `03.딥러닝_Colab기반_실습환경구축_가이드/` Colab 환경 설정 가이드

## 진행 메모
- 2026-09-14: 보드 6번(10.10.15.61) SSH 접속 성공. `work/samples/`에 강사 샘플 2개 있음. sample_01(MobileNetV2 분류 224, MediaPipe로 손 crop) 약 15.8 FPS, sample_02(YOLO 320, float32, NMS 직접 구현) 약 12 FPS. 분류 모델은 배경/손 위치에 민감해 인식률 낮음. 다음: `02.examples(Board)/examples`를 `work/`에 넣고 EX_03에서 best / best_w8a32 / best_int8 FPS 비교.
- 2026-09-14: Colab `EX_01_Colab_Webcam`(OpenCV 5 때문에 contrib 설치 + haar xml wget 필요), `EX_03_RPS_MobileNetV2` 실행. 보드 `EX_02_Board_Webcam` 성공(14~27 FPS). 학습노트 `학습노트/MobileNetV2_RPS_학습노트.html` 작성(EX_03 한 줄씩).
- 2026-09-15: 학습노트 `학습노트/Board_LiteRT_EX04_학습노트.html` 작성(EX_04 한 줄씩 + EX_05/07 차이). EX_04 30줄은 RGB 변환 결과 대신 원본 frame을 resize하는 실수가 있음(EX_05는 정상).
- 2026-09-15: 보드 EX_04 실행 성공(강사 모델). 약 27.7 FPS. 손이 없어도 'rock'을 표시함(화면 전체 분류의 한계 확인). 예제는 반드시 해당 폴더로 cd 후 실행(모델 경로가 상대경로). 내 tflite는 Drive→PC 다운로드→MobaXterm 드래그로 넣는 방법 안내함. EX_04 30줄 frame→img로 고치니 인식이 조금 좋아짐(BGR 실수 영향 확인). 가위가 가장 약함(화면 전체 축소 시 손가락 틈이 뭉개짐). 보드의 원본은 `EX_04_backup.py`로 백업.
- 2026-09-15: 학습노트 `학습노트/Board_HandDetect_EX05_학습노트.html` 작성(EX_05 한 줄씩, make_square_img 숫자 예시 포함).
- 2026-09-15: RPS_Dataset은 거의 오른손(손바닥 위주, 일부 손등), 배경은 회색/하늘색/흰 벽. 왼손은 EX_03 모델(좌우반전 증강 없음)에서 약함. EX_06은 horizontal_flip=True라 EX_07 모델이 왼손에 강할 것으로 기대.
- 2026-09-15: 학습노트 `학습노트/Augmentation_EX06_학습노트.html` 작성(EX_06 한 줄씩, 증강 데모 그림 포함). 참고: EX_06의 rotation_range=0.2는 단위가 도(°)라 사실상 회전 효과 없음.
- 2026-09-15: 학습노트 `학습노트/Board_Augmented_EX07_학습노트.html` 작성(EX_07, EX_05와 10줄만 다름, 세 모델 비교 실험표 포함). EX_05에서 draw=True로 관절을 그리면 잘라낸 사진에 선이 들어가 오답 증가 확인. 03_CNN 장 노트 6개 완료(EX_03/04/05/06/07 + 크랙).
- 2026-09-15: `학습노트/개념문답_학습노트.html`에 밀린 문답 일괄 추가. E장에 .flow()/제너레이터, 프루닝 vs 드롭아웃 카드. 새 장 I(교육장 보드 실습 문답 10개), J(가위바위보 모델 성능 이해 9개). 앞으로 개념 질문은 바로바로 이 노트에 추가할 것.
- 2026-09-15: 사용자 측정 EX_05 약 27 FPS vs EX_07 약 15.6 FPS. 코드·모델 계산량이 같으므로 손 유무 차이(손 없으면 `if not hands: return`으로 모델 생략)로 해석. 손 넣은 상태로 재측정 필요. 개념문답에 프루닝/클러스터링/양자화 비교, BGR 실수의 실제 영향, FPS 함정 카드 추가.
- 2026-09-15 (다른 폴더 세션에서 진행, 여기로 통합): 교수님 1기 포트폴리오 피드백 PDF → `참고/`로 이동, `포트폴리오_작성규칙.md`·`포트폴리오_트러블슈팅_기록.md`(TS-01~07) 작성. `학습노트/Lightweighting_학습노트.html`(04_Lightweighting 13개 노트북, EX_01 전체 한 줄씩) 작성. EX_01/02 실행 결과: test 0.9906 vs 0.9908(동률), baseline이 0.9900 vs 0.9911로 달라 비교 무효(EX_01 학습 셀 재실행으로 파일 덮어씀). 다음: EX_01 한 번만 돌려 baseline 고정 후 EX_02 재비교, final_sparsity 0.99 실험. `산학_RnD_최종제출자료/`는 팀원 자료 포함이라 .gitignore 처리(저장소 public).
- 2026-09-15: 교수님 피드백 PDF 38쪽을 이 세션에서 직접 전부 확인(글자 추출 + 페이지 렌더링). `포트폴리오_작성규칙.md`에 Ⅳ장(그림에서만 보이는 레이아웃·우리 실습 대응·TS 레퍼런스 실물 패턴) 추가. 핵심: 잘한 TS 3건은 전부 같은 종류 화면을 Before/After 두 번 캡처.
- 2026-09-15: 학습노트 전수 확인. 작성 주체 — Git(9/9 이전 세션), CNN_Crack·MobileNetV2_RPS·EX04·EX05·EX06·EX07(이 세션), Lightweighting(다른 폴더 세션), 개념문답(집 노트북 세션 0~H장 + 이 세션 E장 3카드·I·J장 + 다른 폴더 세션 7카드). 개념문답 D장에 교육장 PC 경로 안내 카드 추가(경로가 집 노트북 기준이라).

## 이번 주 실습: On-Device AI (라즈베리파이5 + YOLO 가위바위보) — 2026-09-14 추가
집 노트북 Claude의 인수인계 메모를 이 PC 경로 기준으로 옮긴 것. 원본 전체는 `교육장_대화전체_이전/04_작업파일/CLAUDE.md` (git 제외 폴더).

### 자료 위치 `Lec_python/2/` (40GB. .img / .tflite / zip / exe는 .gitignore로 제외)
- `2/260316_RPI5_Image_32GB/RPI5_Image_32GB_260316a.img` : 보드용 SD카드 이미지(32GB). win32diskimager로 굽는다.
- `2/On-DeviceAI라즈베리파이5_Lab_[64-302-01]/`
  - `01.examples(COLAB)/examples(COLAB)/` : `03_CNN…`(MobileNetV2 분류), `04_Lightweighting/`(PTQ·QAT 경량화), `05_Object_Detection…/EX_02_RPS_Pretrained_YOLO_Finetuning.ipynb`
  - `02.examples(Board)/examples/05_Object_Detection_Based_On-Device_AI/` : 보드용 `EX_03_Board_RPS_PreTrained_YOLO.py` + `best.tflite`, `best_int8.tflite`, `best_w8a32.tflite`
  - `05.딥러닝_보드_실습환경구축_가이드/딥러닝_보드_실습환경구축_가이드_V120.pdf` : 가이드 기준 보드 IP 192.168.10.3, PC IP 192.168.10.2 (PC-보드 직결 시). MobaXterm으로 SSH 접속, 계정·비밀번호는 가이드 PDF 참고, 예제는 보드의 /home/willtek/work/
  - **교육장 실제 값 (2026-09-14)**: 보드와 PC 모두 교육장 네트워크 10.10.15.0/24에 연결. 보드 IP **10.10.15.61**, PC IP 10.10.15.115. 가이드의 192.168.10.x 설정은 하지 않는다. 보드 IP는 이미 잡혀 있어서 전원만 켜면 SSH 접속됨(확인 완료). 교육장 보드 6번은 LCD 없는 맨 라즈베리파이5 + 냉각팬. MobaXterm은 `05.딥러닝_보드_실습환경구축_가이드/01.MobaXterm/MobaXterm_Portable_v11.1/MobaXterm_Personal_11.1.exe`.
  - `03.기타자료/` : LabelImg(라벨링 도구), Netron(모델 구조 보기)
- `2/Deep_Learning(축약)_Lab_[64-301-01]/02.DL(CNN)_Files/save/rps_yolo11n.onnx` : 같은 가위바위보 모델의 ONNX판. 입력 `images` [1,3,320,320], 출력 `output0` [1,7,2100], 클래스 0=scissors 1=rock 2=paper
- `2/ON-Device AI를 위한 Deep Learning/` : 위 내용과 겹치는 복사본. 나중에 정리.

### 확인된 사실
- 실습 보드는 윌텍(Willtek)이 라즈베리파이 5 Model B로 만든 보드(LCD, USB 웹캠). 과정 장비 목록의 TOPST가 아니다.
- `yolo11n.pt`는 폴더에 없다. Colab에서 학습 명령을 치면 Ultralytics가 자동 다운로드한다.
- EX_03 코드는 손 모양 인식까지만 있고 **승패 판정 로직은 없다** (직접 추가할 부분).
- 교육 자료에 PC/ONNX용 추론 코드는 없다. 집에서 만든 것이 아래 폴더.

### `Lec_python/집에서_미리해보기/` (집 노트북에서 만든 PC 웹캠판)
- `EX_03_PC_Webcam_RPS_YOLO_ONNX.py` : EX_03을 onnxruntime + PC 웹캠용으로 바꾼 것. 바꾼 줄에 `[변경]` 주석.
  - ONNX 출력 좌표는 이미 0~320 픽셀 단위라, tflite판처럼 `× IMG_SIZE`를 하면 안 된다.
  - 모델 경로는 `../2/Deep_Learning*/02.DL(CNN)_Files/save/rps_yolo11n.onnx`를 glob으로 찾는다. 이 PC 구조에서 그대로 맞는다.
- 실행 전 가상환경 필요: `python -m venv .venv` → `.venv\Scripts\python.exe -m pip install onnxruntime opencv-python numpy`
- `실행.bat`은 ASCII만 사용(한글 넣으면 cmd에서 깨짐). 웹캠 창 실제 동작은 아직 확인 전.
- (2026-09-14 교육장 PC) `.venv` 생성 및 onnxruntime 1.30 / opencv 5.0 / numpy 2.4 설치 완료. 모델 로딩 확인. 웹캠은 0번(모니터 내장 "LG AIO MNT").
  - 처음엔 `import onnxruntime`에서 접근 위반으로 죽었음. 원인은 이 PC의 시스템 VC++ 런타임이 14.14(2018)로 오래돼서. 관리자 권한이 없어 재배포 패키지 설치 대신, OneDrive 폴더에 있던 14.50짜리 `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`을 `%LOCALAPPDATA%\Programs\Python\Python311\`에 복사해서 해결(원본은 `.bak_14.34`로 백업). 가상환경의 python.exe는 껍데기라 본체 폴더에 넣어야 함.

### 2026-09-15 (오후) 04_Lightweighting 학습노트 재작성
- `학습노트/Lightweighting_학습노트.html`을 이 세션이 처음부터 다시 만들었다(13개 노트북 코드 한 줄씩 + 실행 결과·그림 5장 포함, 26절). 다른 폴더 세션이 만든 개념 정리판은 오류 3곳을 정정해 `학습노트/구버전/`으로 옮겼다.
- 확인 도구: `Lec_python/집에서_미리해보기/.venv`에 `tflite`·`flatbuffers`(flatbuffer 파서) 설치. `ai-edge-litert`는 설치는 되나 이 PC에서 DLL 로드 실패라 안 씀. 스크립트는 세션 scratchpad `inspect_tflite2.py`.
- 검증된 핵심 사실: 보드 자료의 `RPS_MobileNetV2_Augmentation_QAT.tflite`(2,883,168 B, int8 입출력)는 EX_14 노트북 그대로의 출력(4,880,152 B, float 입출력 = `…_QAT_Before.tflite`)이 아니라 주석 처리된 INT8 블록으로 만든 것. EX_15는 int8 입출력만 전제(float이면 84줄 0 나누기). → TS-08.
- EX_10·EX_11 `print_model_weights_sparsity`는 정상(`np.count_nonzero(weight == 0)`). TS-07의 해당 항목 삭제.
- 개념문답 K장 추가: 대학원 산학 프로젝트 코드(utils/preprocess·postprocess 분리, asyncio 큐, 별도 프로세스 표시)와 실습 코드(단일 파일 processImage) 비교.
- 2026-09-16: 사용자 지적("생략한 게 많아")으로 노트를 노트북 JSON 기반 생성 방식으로 재작성. 13개 노트북 코드 셀 299개 전부 원문 그대로 + 실행 결과 + 설명. 완전성은 scratchpad `check_complete.py`로 기계 확인(0 누락). 생성기 `gen_note.py` + 설명 `expl_a.py`/`expl_b.py`.
- `.gitignore`에 `백업4_이력서·대화·메모리_막판/` 추가(이력서 등 개인 문서. 이전엔 untracked 상태로 노출 위험).

### 2026-09-16 05_Object_Detection 학습노트 작성
- `학습노트/ObjectDetection_YOLO_학습노트.html` 신규(24절). EX_01(32줄)·EX_02(코드 셀 9개)·EX_03(170줄) 코드 전부 원문 + Colab 실행 결과 + 설명. 완전성은 스크립트로 기계 확인(누락 0).
- 이번 주 핵심 전환: **분류 → 검출**. YOLO가 위치까지 찾으므로 MediaPipe 손 검출이 필요 없다. 대신 후처리(필터링+NMS+좌표 복원)가 약 40줄로 늘어난다.
- 확인한 사실: tflite 3개(best 10.55MB / best_int8 3.02MB / best_w8a32 2.92MB) **모두 입출력 float32**. 그래서 EX_03에는 scale/zero 환산 줄이 없고, 모델을 바꿔도 8~10줄 주석만 옮기면 된다. 04의 TS-08 같은 사고가 구조적으로 안 생긴다.
- best_int8은 op가 385로 best(353)보다 많다. 입출력 float32 유지용 Quantize/Dequantize 노드 32개가 끼어서다.
- 데이터셋 RPS_Dataset_YOLO.zip: 학습 82장(박스 100개), 시험 22장(박스 32개). **18/82장은 박스가 2개 이상** — 분류 데이터와 결정적으로 다른 점.
- 학습 결과(강사 실행): mAP50 0.995, mAP50-95 0.862, 80 epoch 1분 52초(T4).
- 미측정: 보드에서 EX_03 세 모델 FPS, benchmark_model 추론 시간. 04의 EX_13(19.2 FPS)과 비교하면 "손 검출 제거 vs 무거운 모델"의 맞바꿈을 수치로 볼 수 있다.

### 2026-09-16 파이프라인 트레이싱 도구 추가
- WARBOY 프로젝트에서 쓰던 Chrome Trace 타임라인을 보드에서도 만들 수 있게 직접 작성했다. Chrome Trace Format은 퓨리오사 전용이 아니라 공개 형식이라 가능.
- 퓨리오사 쪽은 `FURIOSA_PROFILER_OUTPUT_PATH` 환경변수로 켜는 방식이었다(백업 폴더의 `*_tracing.py`에 트레이싱 코드가 없는 이유).
- 새 파일: `02.examples(Board)/examples/trace_util.py`, `04_Lightweighting/EX_13_traced.py`, `05_Object_Detection…/EX_03_traced.py`, `트레이싱_사용법.md`. **원본 예제는 수정하지 않았다**(사본 방식).
- 원본 로직 누락 여부를 스크립트로 대조 확인. 의도적 변경 3곳(early return 분리 2, 모델을 argv로 받기 1)뿐.
- PC에서 동작 확인 완료: JSON 유효, 구간 중첩 정상, psutil 있으면 코어별 사용률 카운터 기록.
- 보드에서 `pip install psutil` 하면 코어 사용률 그래프까지 나온다. 없으면 타임라인만.
- 이걸로 04의 미해결 항목(한 프레임 52 ms 중 추론 외 42.3 ms의 내역)을 확정할 수 있다.

### ▶ 2026-09-17 아침에 이어서 할 것 (9/16 퇴근 시점 정리)

**어제까지 끝난 것**: 04 실습(EX_13) 파이프라인을 트레이싱으로 측정하고 4단계 최적화 완료.
한 프레임 60.54 → 23.19 ms(지연 2.6배 개선). 단, **유효 출력은 카메라 상한 18.5 FPS에 막힘**(중복 57.1%).
상세 기록: `포트폴리오_트러블슈팅_기록.md` M-01 / M-01-B, `학습노트/Lightweighting_학습노트.html` 21절.

현재 `EX_13_opt.py` 설정 (4단계 상태):
```
WAIT_MS = 1 | CAMERA_THREAD = True | DETECT_EVERY = 2 | INTERP_THREADS = 4
```

**1순위 — 카메라 18.5 FPS의 원인 규명**
```
cd ~/work/examples
python3 camera_check.py
```
형식(MJPG/YUYV)·FPS 요청·자동노출을 바꿔 가며 순수 읽기 속도만 5초씩 잰다.
30 FPS 근처가 나오는 조건이 있으면 그 설정을 `EX_13_opt.py`에 넣는다 → 4단계까지의 작업이 전부 실제 성능으로 바뀐다.
어떤 조건에서도 18 근처면 조명을 밝게 하고 재측정(자동노출이 어두우면 프레임률을 떨어뜨린다).

**2순위 — 중복 프레임 건너뛰기 (전력 절감)**
지금은 같은 그림을 57% 다시 처리한다. `EX_13_opt.py`에 이미 중복 감지가 들어 있으니,
중복이면 `processImage`를 건너뛰게 고치면 출력 18.5는 그대로면서 CPU가 절반 이하로 떨어진다.

**3순위 — 남은 실험**
- `DETECT_EVERY = 3`: 어디까지 가고 언제 쓸 수 없어지는지 경계 찾기
- 박스 지연 정량화: 손을 빠르게 흔들었을 때의 인상 기록 (고찰 4 보강용, 아직 미기록)
- 05 검출 파이프라인(`EX_03_traced.py`)에 같은 측정 → 분류 vs 검출 비교

**4순위 — 05 실습 본체**
`학습노트/ObjectDetection_YOLO_학습노트.html`(24절) 만들어 둠. 보드에서 EX_03을 세 모델
(best / best_int8 / best_w8a32)로 돌려 FPS·판정 비교. 손 두 개 넣어 박스 2개 뜨는 것 시연.

**git**: 9/16까지 작업이 커밋 안 돼 있다. 교육장 공용 PC이므로 백업 필요.
`git add -A` → `git status`로 산학/백업4/참고 폴더가 안 올라가는지 확인 → commit → push.

### 2026-09-17 프로젝트(가위바위보 게임) 진행

- `RPS_Game.py` 수정: ① 판정 유예 상태 `JUDGING`(2초) 추가 — 카운트 0인 한 프레임에만 판정하던 것이 11 FPS에서 거의 항상 실패했다. ② 안내 문구 2초 유지(`notice`). ③ `MIRROR=True`로 좌우 반전(`detect()` 앞에서 뒤집어 좌표 일관성 유지). ④ `USE_TOP2`(기본 False) — 얼굴 오검출 임시 방편. ⑤ **MJPG + FPS 30 요청** 추가(강사님 지시). `EX_13_opt.py`, `EX_03_opt.py`에도 같은 설정.
- **데이터셋 실측** (`RPS_Dataset_YOLO` 104장 픽셀 직접 분석): train 82장/박스 100개(가위34 바위33 보33, 1개 64장·2개 18장), test 22장/32개, 전부 640×480. **배경 밝기 V 최소 131 / 평균 153 / 최대 176, V<120인 사진 0장.** 배경 BGR 140/144/146(무채색), 손 BGR 131/139/160(R이 21 높음).
  → "흰 벽 O, 밤색 티 X"의 원인이 수치로 확정됐다. 모델이 배운 규칙은 "밝은 무채색 벽 앞의 붉은 덩어리". 밤색 티는 어둡고 붉어서 단서 둘이 동시에 무너진다. `0 backgrounds`라 얼굴도 손으로 잡는다(뿌리가 같음).
- 새 파일 `05_Object_Detection…/bg_check.py`: 카메라에 보이는 배경의 V·R−B를 1초마다 찍어 학습 범위(131~176, 무채색)와 비교. SSH만으로 동작(창 안 띄움).
- 기록: `포트폴리오_트러블슈팅_기록.md` **TS-09**(얼굴 오검출), **TS-10**(배경 분포 차이 — 이번 프로젝트 최강 소재). 개념문답 **N장 5카드**(데이터 82장의 의미 / 밤색 티 / mAP 0.995의 함정 / 얼굴 / MJPG).
- **다음**: ① 보드에서 `bg_check.py`로 밤색 티와 흰 벽 실측 → TS-10 표 채우기 ② `camera_check.py`로 MJPG 효과 확인 ③ 밤색 티·어두운 배경·얼굴 사진 추가 촬영 + 라벨링 → 재학습 → Before/After 10회 시도 정답률 ④ `EX_03_opt.py` 실행(예측 ~37 ms, ~27 FPS).

### 2026-09-17 (오후) 데이터셋 확장 준비

- **강사님 공지 2건**: ① 05의 `EX_03_Board_RPS_PreTrained_YOLO.py`에서 `tflite_runtime` → `ai_edge_litert`로 교체(같은 물건의 새 이름. 03·04는 처음부터 새 이름이었다). ② 데이터가 xml이라 주의.
- **같은 이름 zip 두 개를 확인했다 (중요)**: `01.examples(COLAB)/files/RPS_Dataset_YOLO.zip`은 **txt 104개 + images/labels 정식 구조**(EX_02가 쓰는 것, 학습 됨). `Deep_Learning(축약)…/02.DL(CNN)_Files/RPS_Dataset_YOLO.zip`은 **xml 104개**(LabelImg 원본, 학습 안 됨). **EX_02 노트북에는 변환 코드가 없다**(코드 셀 9개 전부 확인). LabelImg 기본 저장이 PascalVOC(xml)이라 새로 라벨링하면 안 되는 쪽이 만들어진다.
- 새 파일 `Lec_python/2/데이터셋_확장/make_dataset.py` (PC 실행): xml→txt 변환 + 강사님 82장과 병합 + train/test 분할 + data.yaml + 검사 + zip. **검증: xml 104개를 변환해 강사님 txt와 박스 132개 1:1 대조 → 클래스 불일치 0개, 좌표 최대 차이 0.00064픽셀(반올림 차이).** 새 사진 없이 돌리면 원본(82/100, 34·33·33 / 22/32, 11·8·13)이 그대로 재현된다.
- 새 파일 `05_Object_Detection…/EX_01_Auto_Capture.py` (보드 실행): 1.5초마다 자동 촬영, 직전 저장본과 비슷하면 건너뜀, 1/2/3=가위바위보·4=손2개·**0=배경(라벨 안 만듦)** 태그를 파일명에 기록, 기존 번호 뒤에서 이어 붙임(강사님 사진 덮어쓰기 방지 — 원본 `EX_01_Image_Capture.py`는 항상 img_0001부터 저장한다). 640×480 고정.
- **촬영 계획 60장**: 손2개 30 / 손1개 20 / 배경(손 없음) 10, 그리고 이 60장을 **밤색 티 20 · 어두운 곳 20 · 흰 벽 20**으로 쪼갠다. 현재 train은 손1개 64장(78%)·손2개 18장(22%)인데 게임은 항상 손 2개라 그쪽이 부족하다.
- 라벨링 주의: 클래스명 소문자 `scissors`/`rock`/`paper`, **LabelImg에서 YOLO 형식 직접 저장 금지**(classes.txt 줄 순서로 번호가 정해져 조용히 틀린다 — PascalVOC로 저장 후 변환기 사용), 배경 사진은 라벨 안 만듦, 파일명 충돌 금지.
- 기록: **TS-11**(zip 두 개 + 변환기 검증), 개념문답 카드 4개(ai_edge_litert / 몇 장을 어떻게 찍나 / xml 주의 / 자동 촬영기).
- `.gitignore`에 `데이터셋_확장/captures/`, `RPS_Dataset_YOLO_v2/`, `.zip` 추가.

### 2026-09-18 마감일 아침에 이어서 할 것  (제출 마감 오후 4시)

**호칭 주의**: 이 과정에서 프로젝트를 지도하시는 분은 **교수님**이다(그동안 "강사님"으로 잘못 써 왔다).
제출 보고서와 앞으로의 학습노트에서는 교수님으로 쓴다.

**확보된 핵심 수치** (보고서 4장의 전부). 학습 설정은 한 글자도 바꾸지 않고 데이터만 늘렸다.

| 모델 | 학습 | 쉬운 시험 mAP50 | 쉬운 mAP50-95 | **어려운 mAP50** | **어려운 mAP50-95** |
|---|---|---|---|---|---|
| v1 | 82장 | 0.995 | 0.862 | **0.475** | **0.184** |
| v2 | 327장 | 0.995 | 0.849 | 0.650 | 0.383 |
| v3 | 724장 | 0.995 | 0.827 | **0.851** | **0.472** |

- 시험지 두 개: ① 교수님 원본 test 22장(학습에 미포함, 세 판 공통) ② 새 사진 160장(v3 기준)
- **쉬운 시험지에서는 세 모델이 전부 0.995로 같다. 어려운 시험지에서만 +79%가 드러난다.**
- 보드 실측: 9가지 조합 판정 **v1 3/9 → v3 9/9**. 바위 Recall 0.352 → 0.775.
- 원인: v1 학습 데이터의 바위 박스가 **최대 15.5%**뿐(v3는 40.4%). 개수(33개)는 균형이 맞았으므로 **양이 아니라 크기 다양성**이 문제였다.
- 게임 실행 **68.1 FPS** (MJPG + INT8 + 4스레드). 모델 10.1 MB -> 2.9 MB.

**남은 일 (우선순위)**

1. **v4 학습 마무리** — `Lec_python/2/데이터셋_확장/RPS_Dataset_YOLO_v4.zip`(학습 1071장) 드라이브 업로드 → EX_02 경로 셀을 `_v4`로 → `모두 실행`. 어제 2 epoch에서 중단시켜 실패했으니 **끝까지 둘 것**. export 셀의 `BEST` 자동 탐색을 "epoch이 가장 많은 폴더"로 고칠 것(최근 수정 기준이면 중단된 폴더를 집는다).
2. **보드에서 세 모델 FPS 비교** (15분, 교육장에서만 가능) — `EX_03_traced.py`를 `my_best.tflite` / `my_best_int8_v3.tflite` / `my_best_w8a32.tflite`로 각각 실행. 보고서 3-2장의 빈칸이다.
3. PPT에 1·2번 결과 반영 → **폰트 내장**(파일 > 옵션 > 저장 > "파일에 글꼴 포함") → **PDF 내보내기**.
4. `AI_시스템반도체_설계_2기_최종 결과물(정보윤).zip`으로 묶어 구글 폼 제출. 내용물: 보고서 pptx + pdf, 직접 만든 .py 10종, 시연 영상(oCam 녹화본, 문서 폴더).

**만든 파일**: `AI_시스템반도체_설계_2기_결과보고서(정보윤).pptx`(13장, `make_report_ppt.py`로 생성), `ppt_미리보기.html`, `최종보고서_구성안.md`.
**게임 캡처**: 사진 폴더의 `Screenshots/가위바위보/v1`, `v3` (각 9판 + 터미널 최종 점수).

**주의**: v4는 어려운 시험지가 247장으로 바뀌므로 v3와 점수를 그대로 비교하면 안 된다. 같은 시험지로 v3 모델을 다시 재는 셀(`!yolo val model={save_dir}v3/best.tflite ... split=test`)을 함께 돌려야 진짜 비교가 된다.

**지금 상태로도 제출 가능하다.** v4가 안 되면 v3 수치로, FPS 표가 비면 그 칸만 빼면 된다.

### 다음에 할 만한 것
- 교육장 PC 웹캠으로 실시간 인식 확인
- 보드에서 `best` / `best_int8` / `best_w8a32` FPS·정확도 비교
- EX_03에 두 사람 손의 승패 판정 추가 (화면 좌/우 위치로 사람 구분)
- 화면 출력을 별도 프로세스로 분리해 FPS 개선 (사용자의 대학원 프로젝트 보고서 구조와 같은 방식)

### 참고
- 사용자 대학원 프로젝트: Furiosa WARBOY NPU에서 YOLOv8 객체 탐지 고도화(INT8 PTQ, 후처리 CPU 분리, 멀티프로세스). EX_03의 단일 루프가 그 보고서의 Baseline 구조와 같다. 코드는 AI 도움으로 작성해서 원리부터 다시 설명 필요.
- 지인에게 받은 키트(라즈베리파이5 + M.2 Artix-7 FPGA 카드 + HW571 JTAG)는 집에 있음. **라즈베리파이 전원이 켜진 상태에서 FPGA JTAG의 VCC 핀은 연결하지 않는다.**
