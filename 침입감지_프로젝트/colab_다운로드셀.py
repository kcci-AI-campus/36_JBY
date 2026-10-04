# ============================================================
#  Colab 마지막 셀 — 결과 모아서 내려받기
#  이 파일 내용을 통째로 복사해서 Colab 새 셀에 붙여넣고 실행하세요.
# ============================================================

import shutil, glob, os

os.makedirs('/content/out', exist_ok=True)


def grab(pattern, dest=None, limit=None):
    """어디에 저장됐든 찾아서 /content/out 으로 복사한다."""
    fs = sorted(glob.glob('/content/**/' + pattern, recursive=True))
    fs = [f for f in fs if '/content/out/' not in f]
    if limit:
        fs = fs[:limit]
    for f in fs:
        shutil.copy(f, '/content/out/' + (dest or os.path.basename(f)))
    return len(fs)


print('best.pt   :', grab('weights/best.pt', 'best.pt', 1))
print('tflite    :', grab('*.tflite'))

for p in ['results.png', 'confusion_matrix_normalized.png', 'confusion_matrix.png',
          'PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png',
          'labels.jpg', 'val_batch0_labels.jpg', 'val_batch0_pred.jpg', 'results.csv']:
    grab(p, p, 1)

for f in ['/content/per_class.png', '/content/summary.json']:
    if os.path.exists(f):
        shutil.copy(f, '/content/out/')

print('예측 사진 :', grab('pred/*.jpg', None, 8))

print()
for f in sorted(os.listdir('/content/out')):
    print('%8.2f MB  %s' % (os.path.getsize('/content/out/' + f) / 1048576.0, f))

get_ipython().system('cd /content && zip -qr result.zip out')

from google.colab import files
files.download('/content/result.zip')
