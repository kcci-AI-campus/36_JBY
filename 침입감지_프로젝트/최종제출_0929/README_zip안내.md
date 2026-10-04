# AI 캠퍼스 2기 · 10팀 「지켜봄」 — 결과파일 (제출 항목 4)

팀원: 팀원C · 팀원A · 정보윤 · 팀원B · 묶은 날: 2026-09-29 (정리: 정보윤, 클로드코드로 패키징)

## 구성
| 폴더 | 내용 | 출처 |
|---|---|---|
| `1_팀저장소_Team10_최신본/` | 팀 GitHub 저장소 `kcci-AI-campus/Team10` (Private) 의 main 가지 전체. 커밋 `8c33088` (2026-09-29) | `git archive` 로 그대로 꺼냄 (`__pycache__` 만 제외) |
| `2_추가_정보윤PC/` | 저장소에 올리지 않은 결과 파일 (아래) | 정보윤 PC |

### 1_팀저장소_Team10_최신본/ 안에서 찾을 것
- **소스코드**: `py_codev3/` (최종: `run_demo.py`, `detect_baby_4state.py`, `web.py`, `notify.py`, `clip.py`, `trace_util.py`, `run_experiment.py`), `py_codev2/` (이전 판 + 데이터 도구: 촬영·자동 라벨·분할), `py_codev1/` (초기 웹 스트리밍). 실행 방법은 `py_codev3/README.md`.
- **Colab 노트북(.ipynb)**: `5-model/Colab/` — 5개 모델(YOLO26n · YOLOv8n · YOLOv10n · SSDLite320 · YOLO11n) 학습·성능 비교 6개.
- **Train dataset**: `dataset/final_dataset_v2/` — 학습 1111장 · 시험 227장, 4클래스(baby · adult · knife · outlet), `data.yaml` 포함(`val: images/test`).
- **모델 파일**: `5-model/<모델>/` FP32·INT8 .tflite, `model/` (YOLO11n 512·640 .pt/.tflite, YOLO11s), `py_codev3/models/` (보드 배포본).
- **측정·결과**: `py_codev3/측정결과_0928/` (방식별 전력 5단계, 원본 csv·그래프), `5-model/power_bench/` (모델별 전력), `py_codev2/docs/보드실측_0926/`, `py_codev2/docs/저전력실험_0927/`, `report/` (9/23 추론 성능·전력·INT8 원인 분석), `docs/` (모델 선정 기록·발표 그림), `py_codev3/clips/` (9/28 경보 클립).

### 2_추가_정보윤PC/
| 폴더 | 내용 |
|---|---|
| `Colab_학습노트북/` | `학습_Colab.ipynb` — 위험구역 침입 감지 학습·성능 평가 노트북(실행 출력 포함, 9/22) · `학습_Colab_비교.ipynb` · `셀스크립트/` Colab 에서 쓴 변환·비교 셀 9개 (`colab_*.py`, 이 중 3개는 저장소 `py_codev2/colab/` 에도 있음) |
| `트레이스_0926/` | 움직임 게이팅 끔/켬 60초 트레이스 원본 (`trace_gate_off.json`, `trace_gate_on.json`, Chrome Trace Format · Perfetto 로 열림) + Perfetto 화면 캡처 2장 |
| `교육장_실험_0928/` | 9/28 교육장 실험 계획표(`실험표_0928.md`)와 실행 스크립트(`실험.sh`, `run4.sh`), `결과_0928/` 모델별 검출 점수 비교(md)와 비교 사진 |
| `보드결과_0922-23/` | 9/22 모델 벤치마크(`bench_result.csv`), 9/23 검출률·전력 기록(`detect_rate.csv`, `power_log.json`, `rate_out/`), 9/22 경보 클립 18개 |

## 뺀 것
- `notify.json` (휴대폰 알림 주소가 든 설정 파일) — 저장소와 이 묶음 모두 `notify.json.example` 만 있음.
- 중간 단계 모델 파일(9/23 보드에 있던 by_*, v2_*, jjh_* .tflite)과 실험 입력용 영상 4개(active60 · babyonly60 · mix60 · pan60 .mp4) — 최종 모델과 실측 결과가 위에 있어 제외.
- 발표 자료(ppt·pdf)와 시연 영상은 제출 항목 1~3 으로 따로 제출.
