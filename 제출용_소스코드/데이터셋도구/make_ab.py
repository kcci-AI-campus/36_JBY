# -*- coding: utf-8 -*-
"""
라벨 품질만 다른 두 학습 데이터셋을 만든다.  (A/B 실험)

■ 무엇을 가리려는 것인가

    "라벨을 잘못 치면 학습에 나쁜가"를 재려면, **라벨 말고는 전부 같아야** 한다.
    지금까지 비교했던 판들은 사진도 다르고 장수도 달라서, 차이가 나도
    라벨 때문인지 사진 때문인지 구분할 수 없었다.

    여기서는 **같은 사진 88장**에 붙은 두 벌의 라벨만 바꾼다.
      A : 팀원이 붙인 라벨 그대로
      B : 같은 사진을 내가 다시 라벨링한 것

    두 벌의 차이는 실측했다 (2026-09-18, 88장 대조)
      · 클래스가 다른 것 0장  ← 가위/바위/보 자체는 팀원도 정확했다
      · IoU 평균 0.693, IoU<0.5 가 17장(19%)
      · 팀원 박스가 평균 52% 크다. 특히 '보'는 12.9% vs 26.5% 로 2배
        (팔뚝·책·책상까지 감싼 박스가 다수)

■ 나머지 학습 데이터 (A, B 공통)

    교수님 원본 train 82장  +  2026-09-18 직접 촬영 두 손 87장(배경 10장 포함)
    ※ 시험지로 떼어 둔 48장(holdout.txt)은 두 판 모두에서 제외

■ 채점

    hold40 48장(박스 86개)  — A, B 어느 쪽도 학습에 쓰지 않았고 조건이 대등하다
    교수님 22장(박스 32개)  — 참고용. 박스가 적어 차이를 읽기 어렵다
"""
import io
import os
import glob
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
LEC = os.path.abspath(os.path.join(HERE, '..', '..'))      # Developer/Lec_python
ORIG_ZIP = os.path.join(
    HERE, '..', 'On-DeviceAI라즈베리파이5_Lab_[64-302-01]',
    '01.examples(COLAB)', 'files', 'RPS_Dataset_YOLO.zip')
CAP = os.path.join(HERE, 'captures')
RE = os.path.join(LEC, '팀원사진_재라벨')
HOLD = set(l.strip() for l in io.open(os.path.join(HERE, 'holdout.txt'),
                                      encoding='utf-8') if l.strip())
CLASSES = ['scissors', 'rock', 'paper']


def build(name, use_mine):
    out = os.path.join(HERE, 'RPS_Dataset_YOLO_' + name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    for sp in ('train', 'test'):
        os.makedirs(os.path.join(out, 'images', sp))
        os.makedirs(os.path.join(out, 'labels', sp))

    n = {'교수님': 0, '내 두손': 0, '팀원 88장': 0, '배경': 0, '박스': 0}

    # 1) 교수님 원본 (train 82 / test 22 를 그대로)
    z = zipfile.ZipFile(ORIG_ZIP)
    for e in z.namelist():
        p = e.split('/')
        if len(p) < 4 or p[1] not in ('images', 'labels') or p[2] not in ('train', 'test'):
            continue
        if not p[3]:          # zip 안의 폴더 항목은 건너뛴다
            continue
        with open(os.path.join(out, p[1], p[2], p[3]), 'wb') as f:
            f.write(z.read(e))
        if p[1] == 'images' and p[2] == 'train':
            n['교수님'] += 1

    # 2) 내가 찍은 두 손 (시험지로 뺀 것 제외)
    for d in sorted(os.listdir(CAP)):
        p = os.path.join(CAP, d)
        if not os.path.isdir(p):
            continue
        for j in sorted(glob.glob(os.path.join(glob.escape(p), '*.jpg'))):
            base = os.path.splitext(os.path.basename(j))[0]
            if base in HOLD:
                continue
            t = j[:-4] + '.txt'
            if not os.path.exists(t):
                continue
            lines = [l.strip() for l in io.open(t, encoding='utf-8')
                     if len(l.split()) == 5]
            stem = 'cap_%s__%s' % (d, base)
            shutil.copy2(j, os.path.join(out, 'images', 'train', stem + '.jpg'))
            if lines:
                io.open(os.path.join(out, 'labels', 'train', stem + '.txt'),
                        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
                n['박스'] += len(lines)
            else:
                n['배경'] += 1
            n['내 두손'] += 1

    # 3) 팀원 사진 88장 — 라벨만 A/B 로 갈린다
    man = {}
    for ln in io.open(os.path.join(RE, '목록.txt'), encoding='utf-8'):
        c, b, t = ln.rstrip('\n').split('\t')
        man[b] = (c, t)
    for b, (c, their_txt) in sorted(man.items()):
        j = os.path.join(RE, c, b)
        # 목록.txt 에 적힌 팀원 라벨 경로는 Lec_python 기준 상대경로다
        if use_mine:
            src_txt = j[:-4] + '.txt'
        elif os.path.isabs(their_txt):
            src_txt = their_txt
        else:
            src_txt = os.path.join(LEC, their_txt)
        lines = [l.strip() for l in io.open(src_txt, encoding='utf-8')
                 if len(l.split()) == 5]
        stem = 'mate__' + os.path.splitext(b)[0]
        shutil.copy2(j, os.path.join(out, 'images', 'train', stem + '.jpg'))
        io.open(os.path.join(out, 'labels', 'train', stem + '.txt'),
                'w', encoding='utf-8').write('\n'.join(lines) + '\n')
        n['팀원 88장'] += 1
        n['박스'] += len(lines)

    with io.open(os.path.join(out, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write('# A/B 실험 - 사진은 같고 팀원 88장의 라벨만 다르다\n')
        f.write('#   %s\n\n' % ('B: 내가 다시 라벨링' if use_mine else 'A: 팀원 라벨 그대로'))
        f.write('path: /content/RPS_Dataset_YOLO_%s\n' % name)
        f.write('train: images/train\n')
        f.write('val: images/test\n\nnames:\n')
        for i, c in enumerate(CLASSES):
            f.write('  %d: %s\n' % (i, c))

    zp = out + '.zip'
    if os.path.exists(zp):
        os.remove(zp)
    zf = zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED)
    for r, _, files in os.walk(out):
        for fn in files:
            full = os.path.join(r, fn)
            zf.write(full, os.path.relpath(full, HERE).replace(os.sep, '/'))
    zf.close()

    tot = len(glob.glob(os.path.join(out, 'images', 'train', '*.jpg')))
    print('%-4s 학습 %d장 (교수님 %d + 내 두손 %d + 팀원 %d, 배경 %d)  박스 %d개  %.1f MB'
          % (name, tot, n['교수님'], n['내 두손'], n['팀원 88장'], n['배경'],
             n['박스'], os.path.getsize(zp) / 1048576.0))


build('A', use_mine=False)   # 팀원 라벨
build('B', use_mine=True)    # 내가 다시 라벨링
print()
print('두 판의 사진은 완전히 동일하다. 팀원 88장의 라벨만 다르다.')
