# -*- coding: utf-8 -*-
"""
새로 찍어 직접 라벨링한 사진으로 **고정 시험지**를 만든다.  (PC에서 실행)

■ 왜 따로 만드는가

    make_dataset.py 는 새 사진을 섞어서 20%를 test_hard 로 뽑는다.
    그런데 섞는 대상이 SOURCES 전체라, 소스를 하나만 바꿔도 시험지 구성이 바뀐다.
    시험지가 매번 달라지면 v1·v3·v4·v6 의 점수를 나란히 놓고 비교할 수 없다.

    이 시험지는 **한 번 만들고 다시 만들지 않는다.** 그래서 어느 모델에 돌려도
    같은 잣대가 된다. 게다가 이 135장은 v1·v3·v4 가 학습에 쓴 적이 없다.

■ 구성

    captures/  (보드에서 직접 촬영, LabelImg 로 손수 라벨링)
        paper_paper … scissors_scissors   두 손이 든 사진 120장 / 박스 240개
        _bg                                손이 없는 배경 사진 15장

    배경 사진을 일부러 넣는다. 얼굴이나 벽을 손으로 잡는 오검출(False Positive)이
    점수에 반영되게 하기 위해서다. 손만 있는 시험지로는 그것을 못 잰다.

■ 쓰는 법

    python make_testset.py
      -> RPS_TestSet_real.zip 이 만들어진다

    구글 드라이브 files/ 에 올리고 Colab 에서

        !unzip -q -o /content/drive/MyDrive/files/RPS_TestSet_real.zip -d /content
        !yolo val model=<모델경로> data=/content/RPS_TestSet_real/data.yaml imgsz=320
"""
import io
import os
import glob
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, 'captures')
OUT_DIR = os.path.join(HERE, 'RPS_TestSet_real')
OUT_ZIP = OUT_DIR + '.zip'

CLASSES = ['scissors', 'rock', 'paper']

if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(os.path.join(OUT_DIR, 'images', 'val'))
os.makedirs(os.path.join(OUT_DIR, 'labels', 'val'))

n_img = n_box = n_bg = 0
per_class = dict((c, 0) for c in CLASSES)
bad = []

for d in sorted(os.listdir(CAP)):
    p = os.path.join(CAP, d)
    if not os.path.isdir(p):
        continue
    want = None if d == '_bg' else sorted(d.split('_'))
    for img in sorted(glob.glob(os.path.join(glob.escape(p), '*.jpg'))):
        base = os.path.splitext(os.path.basename(img))[0]
        txt = img[:-4] + '.txt'
        if not os.path.exists(txt):
            bad.append('%s/%s : 라벨 파일이 없다' % (d, base))
            continue

        lines = []
        for ln in open(txt):
            q = ln.split()
            if len(q) != 5:
                continue
            cid = int(q[0])
            vals = [float(x) for x in q[1:]]
            if not (0 <= cid < len(CLASSES)):
                bad.append('%s/%s : 클래스 번호 %d' % (d, base, cid))
                break
            if not all(0.0 <= v <= 1.0 for v in vals):
                bad.append('%s/%s : 좌표가 0~1 밖' % (d, base))
                break
            if vals[2] <= 0 or vals[3] <= 0:
                bad.append('%s/%s : 박스 크기가 0' % (d, base))
                break
            lines.append('%d %.6f %.6f %.6f %.6f' % tuple([cid] + vals))
        else:
            # 폴더 이름과 맞는지 마지막으로 한 번 더 본다
            got = sorted(CLASSES[int(l.split()[0])] for l in lines)
            if want is not None and got != want:
                bad.append('%s/%s : 라벨이 %s (폴더는 %s)'
                           % (d, base, '+'.join(got), '+'.join(want)))
                continue
            if want is None and lines:
                bad.append('%s/%s : 배경 사진인데 박스가 있다' % (d, base))
                continue

            out = '%s__%s' % (d, base)
            shutil.copy2(img, os.path.join(OUT_DIR, 'images', 'val', out + '.jpg'))
            with open(os.path.join(OUT_DIR, 'labels', 'val', out + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            n_img += 1
            if lines:
                n_box += len(lines)
                for l in lines:
                    per_class[CLASSES[int(l.split()[0])]] += 1
            else:
                n_bg += 1

if bad:
    print('!! 문제가 있어 뺀 사진 %d장' % len(bad))
    for b in bad[:20]:
        print('   -', b)
    raise SystemExit('\n고치고 다시 돌리세요. 시험지에 틀린 정답이 들어가면 '
                     '모든 비교가 무의미해집니다.')

with io.open(os.path.join(OUT_DIR, 'data.yaml'), 'w', encoding='utf-8') as f:
    f.write('# 고정 시험지 - 한 번 만들고 다시 만들지 않는다\n')
    f.write('#   v1, v3, v4 가 학습에 쓴 적이 없는 사진들이다\n')
    f.write('#   재는 법: yolo val model=<모델> data=data.yaml imgsz=320\n\n')
    f.write('path: /content/RPS_TestSet_real\n')
    f.write('train: images/val\n')
    f.write('val: images/val\n')
    f.write('\nnames:\n')
    for i, c in enumerate(CLASSES):
        f.write('  %d: %s\n' % (i, c))

if os.path.exists(OUT_ZIP):
    os.remove(OUT_ZIP)
zf = zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED)
for root, _, files in os.walk(OUT_DIR):
    for fn in files:
        full = os.path.join(root, fn)
        zf.write(full, os.path.relpath(full, HERE).replace(os.sep, '/'))
zf.close()

print('시험지 만들었다')
print('  사진      %d장  (손 있음 %d장 / 배경 %d장)' % (n_img, n_img - n_bg, n_bg))
print('  박스      %d개' % n_box)
for c in CLASSES:
    print('    %-9s %d개' % (c, per_class[c]))
print('  zip       %s  (%.1f MB)' % (OUT_ZIP, os.path.getsize(OUT_ZIP) / 1048576.0))
