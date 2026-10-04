# -*- coding: utf-8 -*-
"""
2026-09-18 에 직접 촬영하고 손수 라벨링한 사진 전부를 **하나의 시험지**로 묶는다.

■ 왜 이 시험지가 필요한가

    지금까지 쓰던 시험지 두 개는 각각 한계가 있었다.
      · 교수님 22장  : 박스가 32개뿐이고 v1·v2·v3·v9 가 전부 mAP50 0.995 라 구분이 안 된다
      · hold40 48장  : v6·v7 은 같은 세션 사진을 학습에 써서 유리하다

    이 시험지는 **학습에 한 장도 쓰지 않은** 사진으로만 구성한다.
    v1 · v2 · v3 · v9 네 판 모두에게 공정하고, 박스가 328개라 훨씬 덜 흔들린다.

    주의: v6 와 v7 은 이 사진 중 87장을 학습에 썼으므로 이 시험지로 채점하면 안 된다.

■ 구성

    두 손 사진 120장 (9조합)      박스 240개   ← captures/
    손 1개 사진  88장 (3클래스)    박스  88개   ← 손1개/
    배경 사진    25장              박스   0개   ← 얼굴·벽 오검출을 점수에 반영
    ------------------------------------------------
    합계        233장             박스 328개

■ 쓰는 법

    python make_testset_today.py
      -> RPS_TestSet_today.zip

    Colab 에서
        !unzip -q -o /content/drive/MyDrive/files/RPS_TestSet_today.zip -d /content
        !yolo val model=<모델> data=/content/RPS_TestSet_today/data.yaml imgsz=320
"""
import io
import os
import glob
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [
    (os.path.join(HERE, 'captures'), 'two'),                       # 두 손 + 배경
    (os.path.join(HERE, '..', '..', 'Lec_python', '손1개'), 'one'),  # 손 1개 + 배경
]
# 손1개 폴더는 Developer/Lec_python/손1개 이므로 상대 경로를 바로 잡는다
SRC[1] = (os.path.abspath(os.path.join(HERE, '..', '..', '손1개')), 'one')

OUT_DIR = os.path.join(HERE, 'RPS_TestSet_today')
OUT_ZIP = OUT_DIR + '.zip'
CLASSES = ['scissors', 'rock', 'paper']

if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(os.path.join(OUT_DIR, 'images', 'val'))
os.makedirs(os.path.join(OUT_DIR, 'labels', 'val'))

n_img = n_box = n_bg = 0
bad = []
per = {}

for root, kind in SRC:
    if not os.path.isdir(root):
        raise SystemExit('폴더가 없다: %s' % root)
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        # 폴더 이름이 곧 정답. _bg 는 손이 없는 사진
        want = None if d == '_bg' else sorted(d.split('_'))
        for img in sorted(glob.glob(os.path.join(glob.escape(p), '*.jpg'))):
            base = os.path.splitext(os.path.basename(img))[0]
            txt = img[:-4] + '.txt'
            if not os.path.exists(txt):
                bad.append('%s/%s : 라벨 없음' % (d, base))
                continue
            lines = [l.strip() for l in io.open(txt, encoding='utf-8')
                     if len(l.split()) == 5]
            got = sorted(CLASSES[int(l.split()[0])] for l in lines)
            if want is not None and got != want:
                bad.append('%s/%s : 라벨 %s (폴더는 %s)'
                           % (d, base, '+'.join(got) or '빈파일', '+'.join(want)))
                continue
            if want is None and lines:
                bad.append('%s/%s : 배경인데 박스가 있다' % (d, base))
                continue

            out = '%s_%s__%s' % (kind, d, base)
            shutil.copy2(img, os.path.join(OUT_DIR, 'images', 'val', out + '.jpg'))
            with io.open(os.path.join(OUT_DIR, 'labels', 'val', out + '.txt'),
                         'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            n_img += 1
            per[kind] = per.get(kind, 0) + 1
            if lines:
                n_box += len(lines)
            else:
                n_bg += 1

if bad:
    print('!! 문제가 있어 뺀 사진 %d장' % len(bad))
    for b in bad[:20]:
        print('   -', b)
    raise SystemExit('\n시험지에 틀린 정답이 들어가면 모든 비교가 무의미해집니다. 고치고 다시 돌리세요.')

with io.open(os.path.join(OUT_DIR, 'data.yaml'), 'w', encoding='utf-8') as f:
    f.write('# 2026-09-18 직접 촬영 + 손수 라벨링. 학습에 한 장도 쓰지 않았다.\n')
    f.write('#   v1, v2, v3, v9 의 공통 잣대 (v6, v7 은 일부를 학습에 썼으므로 제외)\n\n')
    f.write('path: /content/RPS_TestSet_today\n')
    f.write('train: images/val\n')
    f.write('val: images/val\n')
    f.write('\nnames:\n')
    for i, c in enumerate(CLASSES):
        f.write('  %d: %s\n' % (i, c))

if os.path.exists(OUT_ZIP):
    os.remove(OUT_ZIP)
zf = zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED)
for r, _, files in os.walk(OUT_DIR):
    for fn in files:
        full = os.path.join(r, fn)
        zf.write(full, os.path.relpath(full, HERE).replace(os.sep, '/'))
zf.close()

print('시험지 완성')
print('  두 손 쪽 %d장 / 손 1개 쪽 %d장' % (per.get('two', 0), per.get('one', 0)))
print('  사진 %d장  (손 있음 %d장 / 배경 %d장)' % (n_img, n_img - n_bg, n_bg))
print('  박스 %d개' % n_box)
print('  zip  %s  (%.1f MB)' % (OUT_ZIP, os.path.getsize(OUT_ZIP) / 1048576.0))
