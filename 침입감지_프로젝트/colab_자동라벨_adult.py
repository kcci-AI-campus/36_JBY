# ============================================================
#  Colab — adult 사진 자동 라벨링
#
#  COCO 사전학습 YOLO 는 'person' 을 아주 잘 찾는다.
#  그 박스를 그대로 쓰고 클래스 번호만 1(adult) 로 바꿔 저장한다.
#
#  올릴 파일 :
#    C:\Users\kccistc\Developer\침입감지_프로젝트\adult_photos.zip
#
#  결과 : adult_labels.zip 이 자동으로 내려받아진다.
# ============================================================

get_ipython().system('pip -q install "ultralytics==8.4.98"')

import os, glob, shutil, zipfile
from google.colab import files
from ultralytics import YOLO

# 1) 사진 올리기
up = files.upload()                      # adult_photos.zip 선택
for z in up:
    get_ipython().system(f'unzip -q -o "{z}" -d /content/')

SRC = '/content/adult'
OUT = '/content/adult_labels'
os.makedirs(OUT, exist_ok=True)
imgs = sorted(glob.glob(SRC + '/*.jpg'))
print('사진 %d장' % len(imgs))

# 2) COCO 모델로 사람 찾기
#    conf 를 낮게(0.25) 잡는다 — 없는 박스를 새로 그리는 것보다
#    있는 박스를 지우는 편이 훨씬 빠르기 때문이다.
model = YOLO('yolo11n.pt')
res = model.predict(imgs, classes=[0], conf=0.25, imgsz=640,
                    save=True, project='/content', name='preview', exist_ok=True,
                    verbose=False)

# 3) YOLO 형식으로 저장 (클래스 번호를 1 = adult 로)
n_box = 0
n_empty = []
for r in res:
    stem = os.path.splitext(os.path.basename(r.path))[0]
    lines = []
    for b in r.boxes:
        x, y, w, h = b.xywhn[0].tolist()          # 정규화 중심좌표
        lines.append('1 %.6f %.6f %.6f %.6f' % (x, y, w, h))
        n_box += 1
    if not lines:
        n_empty.append(stem)
    open(os.path.join(OUT, stem + '.txt'), 'w').write('\n'.join(lines) + ('\n' if lines else ''))

print()
print('사람 박스 %d개 생성' % n_box)
print('사람을 못 찾은 사진 %d장 :' % len(n_empty), n_empty[:10])

# 4) 박스 그린 미리보기도 같이 담는다 (눈으로 확인용)
for f in glob.glob('/content/preview/*.jpg'):
    shutil.copy(f, os.path.join(OUT, '_preview_' + os.path.basename(f)))

with zipfile.ZipFile('/content/adult_labels.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), f)

print('\n내려받는 중 : adult_labels.zip')
files.download('/content/adult_labels.zip')
