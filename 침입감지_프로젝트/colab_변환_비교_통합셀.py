# ============================================================
#  Colab — tflite 변환(3종) + 정확도 비교 한 번에
#
#  노트북의 "9. tflite 변환" 셀과 그 다음 확인 셀을
#  이 셀 하나로 통째로 바꾸면 됩니다.
#
#  하는 일
#    1) best.pt 를 찾는다 (저장 경로가 어디든)
#    2) format=litert 로 3종 변환  (float32 / INT8 / w8a32)
#       -> format='tflite' 로 내보내면 입력이 NHWC 가 되어 보드 예제와 안 맞는다.
#          format=litert 는 (1,3,320,320) NCHW 로 나온다.
#    3) 5가지를 같은 검증셋으로 평가해 표로 비교
#    4) CSV 저장
# ============================================================

get_ipython().system('pip -q install ai-edge-litert')

import glob, os
import pandas as pd
from ultralytics import YOLO

DATA = '/content/merged/data.yaml'
cands = sorted(glob.glob('/content/**/weights/best.pt', recursive=True))
assert cands, 'best.pt 를 못 찾았습니다. 학습 셀이 끝났는지 확인하세요.'
BEST = cands[0]
WDIR = os.path.dirname(BEST)
print('모델   :', BEST)
print('데이터 :', DATA)

# ── 1) 변환 (이미 있으면 건너뜀) ──────────────────────────
sh = get_ipython().system
jobs = [('best.tflite', ''),
        ('best_int8.tflite', 'quantize=8'),
        ('best_w8a32.tflite', 'quantize=w8a32')]
for fname, opt in jobs:
    if glob.glob(f'{WDIR}/**/{fname}', recursive=True):
        print('이미 있음 :', fname)
        continue
    print('\n>>> 변환 :', fname, opt)
    sh(f'yolo export model={BEST} format=litert imgsz=320 data={DATA} {opt}')

tfl = sorted(glob.glob('/content/**/*.tflite', recursive=True))
print('\n=== 만들어진 tflite ===')
for f in tfl:
    print('  %8.2f MB  %s' % (os.path.getsize(f) / 1048576.0, os.path.basename(f)))
if not tfl:
    raise SystemExit('변환 실패 — 위 로그의 오류를 확인하세요.')

# ── 2) 평가 ───────────────────────────────────────────────
#   best.pt 는 640/320 두 번 재서 "해상도만의 영향"을 분리한다.
targets = [('best.pt (FP32)', BEST, 640),
           ('best.pt (FP32)', BEST, 320)]
for f in tfl:
    targets.append((os.path.basename(f), f, 320))

rows = []
for name, path, sz in targets:
    print('\n' + '=' * 58)
    print(' 평가 :', name, '@', sz)
    print('=' * 58)
    try:
        m = YOLO(path, task='detect')
        r = m.val(data=DATA, imgsz=sz, device='cpu', batch=1,
                  plots=False, verbose=False)
        pc = {}
        for i, ci in enumerate(r.box.ap_class_index):
            pc[r.names[int(ci)]] = round(float(r.box.ap50[i]), 4)
        rows.append({
            '모델': name, '입력': sz,
            '크기(MB)': round(os.path.getsize(path) / 1048576.0, 2),
            '정밀도': round(float(r.box.mp), 3),
            '재현율': round(float(r.box.mr), 3),
            'mAP50': round(float(r.box.map50), 4),
            'mAP50-95': round(float(r.box.map), 4),
            '추론(ms)': round(float(r.speed.get('inference', 0)), 1),
            **{f'AP50_{k}': v for k, v in pc.items()},
        })
        print('  mAP50 %.4f   mAP50-95 %.4f' % (r.box.map50, r.box.map))
    except Exception as e:
        print('  실패 :', e)

df = pd.DataFrame(rows)
print('\n\n' + '=' * 80)
print(' 경량화 비교표   (검증 %s)' % DATA)
print('=' * 80)
print(df.to_string(index=False))

df.to_csv('/content/경량화_비교표.csv', index=False, encoding='utf-8-sig')
os.makedirs('/content/out', exist_ok=True)
df.to_csv('/content/out/경량화_비교표.csv', index=False, encoding='utf-8-sig')
print('\n저장 : /content/out/경량화_비교표.csv')

# ── 3) 기준 대비 하락폭 ───────────────────────────────────
def find(name, sz):
    for r in rows:
        if r['모델'] == name and r['입력'] == sz:
            return r
    return None

pt320 = find('best.pt (FP32)', 320)
fp32 = find('best.tflite', 320)
int8 = find('best_int8.tflite', 320)
w8 = find('best_w8a32.tflite', 320)

print('\n[ 단계별 하락폭 — mAP50-95 ]')
if pt320 and fp32:
    print('  best.pt@320 -> tflite FP32 : %+.4f  (변환 자체의 영향)'
          % (fp32['mAP50-95'] - pt320['mAP50-95']))
if fp32 and int8:
    print('  tflite FP32 -> INT8        : %+.4f  (양자화의 영향)'
          % (int8['mAP50-95'] - fp32['mAP50-95']))
if fp32 and w8:
    print('  tflite FP32 -> w8a32       : %+.4f  (가중치만 8비트)'
          % (w8['mAP50-95'] - fp32['mAP50-95']))
print('\n  ※ mAP50 은 거의 안 떨어지는데 mAP50-95 만 떨어지면,')
print('     "찾기는 찾는데 박스 좌표가 흐트러졌다"는 뜻이다.')
print('     우리 시스템은 박스 사이 거리로 경보를 내므로 이 하락이 판정에 직접 영향을 준다.')
