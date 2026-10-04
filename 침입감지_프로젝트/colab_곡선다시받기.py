# ============================================================
#  빠진 곡선 그래프 4개만 다시 받기
#  Ultralytics 는 검출 작업의 곡선을 Box 접두사를 붙여 저장한다.
#      BoxPR_curve.png  BoxF1_curve.png  BoxP_curve.png  BoxR_curve.png
#  이 파일 내용을 통째로 복사해 Colab 새 셀에 붙여넣고 실행하세요.
# ============================================================

import glob, os, shutil
from IPython.display import Image, display, Markdown

os.makedirs('/content/out', exist_ok=True)

# 실제로 어떤 이름으로 저장됐는지 먼저 확인한다 (버전마다 다를 수 있다)
found = sorted(glob.glob('/content/runs/**/*curve*.png', recursive=True))
print('=== 찾은 곡선 파일 ===')
for f in found:
    print('   ', f)
print()

if not found:
    print('곡선 파일이 없습니다. 학습 폴더 안을 직접 봅니다 :')
    for f in sorted(glob.glob('/content/runs/train640/*')):
        print('   ', f)

# train640 것을 우선으로, 없으면 아무거나 하나씩 가져온다
for f in found:
    name = os.path.basename(f)
    dest = '/content/out/' + name
    if 'train640' in f or not os.path.exists(dest):
        shutil.copy(f, dest)

print('=== /content/out 에 담긴 곡선 ===')
for f in sorted(os.listdir('/content/out')):
    if 'curve' in f.lower():
        print('   ', f)

# 화면에서도 바로 본다
for f in sorted(glob.glob('/content/out/*curve*.png')):
    display(Markdown('### ' + os.path.basename(f)))
    display(Image(filename=f, width=760))

# 다시 묶어서 내려받기
get_ipython().system('cd /content && rm -f result.zip && zip -qr result.zip out')
from google.colab import files
files.download('/content/result.zip')
