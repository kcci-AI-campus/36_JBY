# ============================================================
#  Colab — tflite 변환 + 빠진 그래프 다시 만들기
#  이 내용을 통째로 복사해서 Colab 새 셀에 붙여넣고 실행하세요.
#  (그다음 '다운로드 셀'을 다시 실행하면 tflite 가 담깁니다)
# ============================================================

import glob, os
from ultralytics import YOLO

# 학습된 모델 찾기 (저장 경로가 runs/detect/runs/... 로 들어가 있어도 찾아낸다)
cands = sorted(glob.glob('/content/**/weights/best.pt', recursive=True))
assert cands, 'best.pt 를 못 찾았습니다. 학습 셀이 끝났는지 확인하세요.'
BEST = cands[0]
DATA = '/content/merged/data.yaml'
print('모델 :', BEST)
print('데이터:', DATA)

# ── 1) 평가 그래프 다시 만들기 (PR_curve, F1_curve, P_curve, R_curve) ──
m = YOLO(BEST)
m.val(data=DATA, plots=True)

# ── 2) tflite 변환 3종 — 강사님 EX_02 와 같은 명령 ──
sh = get_ipython().system
sh(f'yolo export model={BEST} format=litert imgsz=320 data={DATA}')
sh(f'yolo export model={BEST} format=litert imgsz=320 data={DATA} quantize=8')
sh(f'yolo export model={BEST} format=litert imgsz=320 data={DATA} quantize=w8a32')

# ── 3) 결과 확인 ──
print()
print('=== 만들어진 tflite ===')
tfl = sorted(glob.glob('/content/**/*.tflite', recursive=True))
for f in tfl:
    print('%8.2f MB  %s' % (os.path.getsize(f) / 1048576.0, f))
if not tfl:
    print('  없음 — 변환이 실패했습니다. 위 로그의 오류 메시지를 확인하세요.')

print()
print('=== 만들어진 그래프 ===')
for p in ['PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']:
    fs = glob.glob('/content/**/' + p, recursive=True)
    print('  %-16s %s' % (p, fs[0] if fs else '없음'))
