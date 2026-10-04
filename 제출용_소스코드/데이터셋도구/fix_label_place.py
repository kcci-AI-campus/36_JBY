# -*- coding: utf-8 -*-
"""
LabelImg 에서 폴더를 바꿀 때 Change Save Dir 을 같이 안 바꾸면
라벨(txt)이 이전 폴더에 계속 쌓인다. 화면에는 아무 경고도 안 뜬다.

이 스크립트는 txt 를 같은 이름의 jpg 가 있는 폴더로 되돌려 놓는다.
사진 이름(brown_two_0124)이 전부 다르기 때문에 어디로 가야 할지 정확히 정해진다.

    python fix_label_place.py            <- 무엇을 옮길지 보여만 준다
    python fix_label_place.py --apply    <- 실제로 옮긴다
"""
import os
import sys
import glob
import shutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captures')
APPLY = '--apply' in sys.argv

# 사진이 어느 폴더에 있는지 먼저 전부 외운다
home = {}
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    for j in glob.glob(os.path.join(p, '*.jpg')):
        home[os.path.splitext(os.path.basename(j))[0]] = p

moved = same = orphan = 0
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    for t in sorted(glob.glob(os.path.join(p, '*.txt'))):
        name = os.path.basename(t)
        if name == 'classes.txt':
            continue
        stem = os.path.splitext(name)[0]
        want = home.get(stem)
        if want is None:
            print('짝 없는 라벨 : %s 안의 %s  (같은 이름 사진이 없다)' % (d, name))
            orphan += 1
        elif os.path.normcase(want) == os.path.normcase(p):
            same += 1
        else:
            print('옮김 : %s 안의 %s  ->  %s 폴더로'
                  % (d, name, os.path.basename(want)))
            if APPLY:
                shutil.move(t, os.path.join(want, name))
            moved += 1

print('-' * 50)
print('제자리 %d개 / 옮길 것 %d개 / 짝 없는 것 %d개' % (same, moved, orphan))
if moved and not APPLY:
    print('실제로 옮기려면 :  python fix_label_place.py --apply')
