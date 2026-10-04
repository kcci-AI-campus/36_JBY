# -*- coding: utf-8 -*-
"""
새로 찍은 사진 중 일부를 **학습에 절대 넣지 않는 시험지**로 떼어 놓는다.

■ 왜 필요한가

    오늘 찍은 135장을 전부 학습에 넣으면 성능은 오르겠지만,
    그 순간 이 사진들은 시험지 자격을 잃는다. 학습에 쓴 사진으로 채점하면
    점수가 부풀려지고 v1·v3·v4 와 비교할 수도 없다.

    그래서 40장을 떼어 둔다. 이 40장은 v1·v3·v4 가 본 적이 없고,
    앞으로 만들 v6 도 보지 않는다. 네 모델의 **공통 잣대**가 된다.

■ 어떻게 고르는가

    폴더(가위바위보 9조합)마다 파일 이름 순으로 3장에 1장씩 뽑는다.
      - 고정된 규칙이라 다시 돌려도 같은 40장이 나온다 (난수 아님)
      - 9조합에서 골고루 뽑히므로 특정 조합만 시험지에 몰리지 않는다
    배경 사진(_bg)도 같은 규칙으로 5장을 뽑는다.
      - 얼굴·벽을 손으로 잡는 오검출을 채점에 반영하기 위해서다

■ 결과

    holdout.txt              떼어 낸 사진 이름 목록 (make_dataset.py 가 읽어서 학습에서 뺀다)
    RPS_TestSet_hold40.zip   그 사진들로 만든 시험지
"""
import io
import os
import glob
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, 'captures')
LIST = os.path.join(HERE, 'holdout.txt')
OUT_DIR = os.path.join(HERE, 'RPS_TestSet_hold40')
OUT_ZIP = OUT_DIR + '.zip'

EVERY = 3        # 손 사진: 3장에 1장씩
EVERY_BG = 3     # 배경 사진: 3장에 1장씩 (15장 -> 5장)

CLASSES = ['scissors', 'rock', 'paper']

picked = []
for d in sorted(os.listdir(CAP)):
    p = os.path.join(CAP, d)
    if not os.path.isdir(p):
        continue
    step = EVERY_BG if d == '_bg' else EVERY
    js = sorted(glob.glob(os.path.join(glob.escape(p), '*.jpg')))
    for i, j in enumerate(js):
        if i % step == 0:
            picked.append((d, j))

with io.open(LIST, 'w', encoding='utf-8') as f:
    for d, j in picked:
        f.write(os.path.splitext(os.path.basename(j))[0] + '\n')

# ── 시험지 zip 만들기 ────────────────────────────────────────
if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(os.path.join(OUT_DIR, 'images', 'val'))
os.makedirs(os.path.join(OUT_DIR, 'labels', 'val'))

n_hand = n_bg = n_box = 0
per_folder = {}
for d, j in picked:
    base = os.path.splitext(os.path.basename(j))[0]
    txt = j[:-4] + '.txt'
    if not os.path.exists(txt):
        raise SystemExit('라벨이 없다: %s' % j)
    lines = [l.strip() for l in io.open(txt, encoding='utf-8') if len(l.split()) == 5]

    want = None if d == '_bg' else sorted(d.split('_'))
    got = sorted(CLASSES[int(l.split()[0])] for l in lines)
    if want is not None and got != want:
        raise SystemExit('라벨이 폴더와 다르다: %s (%s)' % (j, '+'.join(got)))
    if want is None and lines:
        raise SystemExit('배경 사진인데 박스가 있다: %s' % j)

    out = '%s__%s' % (d, base)
    shutil.copy2(j, os.path.join(OUT_DIR, 'images', 'val', out + '.jpg'))
    with io.open(os.path.join(OUT_DIR, 'labels', 'val', out + '.txt'),
                 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))

    per_folder[d] = per_folder.get(d, 0) + 1
    if lines:
        n_hand += 1
        n_box += len(lines)
    else:
        n_bg += 1

with io.open(os.path.join(OUT_DIR, 'data.yaml'), 'w', encoding='utf-8') as f:
    f.write('# 고정 시험지 (holdout) - 학습에 절대 넣지 않는다\n')
    f.write('#   v1, v3, v4, v6 의 공통 잣대\n\n')
    f.write('path: /content/RPS_TestSet_hold40\n')
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

print('시험지로 떼어 낸 사진')
for d in sorted(per_folder):
    print('  %-20s %d장' % (d, per_folder[d]))
print('-' * 40)
print('  손 있는 사진 %d장 (박스 %d개) / 배경 %d장' % (n_hand, n_box, n_bg))
print('  목록  %s' % LIST)
print('  zip   %s  (%.1f MB)' % (OUT_ZIP, os.path.getsize(OUT_ZIP) / 1048576.0))
print()
print('나머지 %d장은 학습에 들어간다.' % (135 - len(picked)))
