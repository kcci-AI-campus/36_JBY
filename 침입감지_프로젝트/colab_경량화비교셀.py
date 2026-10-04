# ============================================================
#  Colab — 모델별 정확도 비교 (해상도 축소 / 양자화 영향 분리)
#  이 내용을 통째로 복사해서 Colab 새 셀에 붙여넣고 실행하세요.
#
#  best.pt @640  ->  best.pt @320  ->  tflite float32 @320  ->  INT8 @320
#  이렇게 한 단계씩 비교해야 정확도가 왜 떨어졌는지 말할 수 있습니다.
# ============================================================

get_ipython().system('pip -q install ai-edge-litert')

import glob, os, json
import pandas as pd
from ultralytics import YOLO

DATA = '/content/merged/data.yaml'
BEST = sorted(glob.glob('/content/**/weights/best.pt', recursive=True))[0]
print('모델:', BEST)

# 비교 대상 : (표시이름, 파일, 입력크기)
targets = [('best.pt (float32)', BEST, 640),
           ('best.pt (float32)', BEST, 320)]
for f in sorted(glob.glob('/content/**/*.tflite', recursive=True)):
    targets.append((os.path.basename(f), f, 320))

rows = []
for name, path, sz in targets:
    print('\n' + '=' * 60)
    print(' 평가 :', name, '@', sz)
    print('=' * 60)
    try:
        m = YOLO(path)
        r = m.val(data=DATA, imgsz=sz, plots=False, verbose=False)
        rows.append({
            '모델': name,
            '입력': sz,
            '크기(MB)': round(os.path.getsize(path) / 1048576.0, 2),
            '정밀도': round(float(r.box.mp), 3),
            '재현율': round(float(r.box.mr), 3),
            'mAP50': round(float(r.box.map50), 4),
            'mAP50-95': round(float(r.box.map), 4),
            '추론(ms)': round(float(r.speed.get('inference', 0)), 1),
        })
    except Exception as e:
        print('  실패:', e)
        rows.append({'모델': name, '입력': sz,
                     '크기(MB)': round(os.path.getsize(path) / 1048576.0, 2),
                     '정밀도': None, '재현율': None,
                     'mAP50': None, 'mAP50-95': None, '추론(ms)': None})

df = pd.DataFrame(rows)
print()
print(df.to_string(index=False))

# 저장
df.to_csv('/content/경량화_비교표.csv', index=False, encoding='utf-8-sig')
print('\n저장: /content/경량화_비교표.csv')

# 기준(첫 줄) 대비 얼마나 떨어졌는지
base = rows[0]
if base['mAP50']:
    print('\n[ best.pt @640 대비 변화 ]')
    for r in rows[1:]:
        if r['mAP50'] is None:
            print('  %-24s 측정 실패' % r['모델'])
            continue
        d = r['mAP50'] - base['mAP50']
        print('  %-24s @%d  mAP50 %.4f  (%+.4f, %+.1f%%p)'
              % (r['모델'], r['입력'], r['mAP50'], d, d * 100))
