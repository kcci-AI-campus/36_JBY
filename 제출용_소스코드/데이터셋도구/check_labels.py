# -*- coding: utf-8 -*-
"""
라벨링 검사기 — LabelImg로 YOLO(txt) 라벨을 친 뒤 바로 돌려서 잘못된 것을 잡는다.

■ 왜 필요한가

YOLO txt에는 클래스가 **번호로만** 들어간다.

    2 0.635156 0.619792 0.339062 0.589583      <- 이 '2'가 무엇인지는 txt에 없다

그 번호의 뜻은 LabelImg가 같이 만드는 **classes.txt의 줄 순서**로 정해진다.
순서가 data.yaml(scissors=0, rock=1, paper=2)과 다르면
**가위를 바위로 배우는데, 오류도 경고도 나지 않는다.** 학습이 다 끝나고서야 안다.

■ 이 검사기가 잡아낼 수 있는 이유

나눠 받은 사진의 **폴더 이름이 곧 정답**이기 때문이다.

    data/paper_rock/   -> 이 안의 사진은 전부 '보 한 손 + 바위 한 손'

그러니 txt에 적힌 번호 두 개가 폴더 이름과 맞는지 **기계가 대조할 수 있다.**
번호가 밀려 있으면(classes.txt 순서 사고) 거의 모든 파일이 한꺼번에 틀리므로 바로 드러난다.

■ 쓰는 법
    python check_labels.py                    # 아래 ROOT 폴더 전체 검사
    python check_labels.py <폴더>             # 폴더 하나만 검사
"""
import os
import sys
import collections

# ─────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', 'data', 'data')   # 나눠 받은 사진이 있는 곳

# data.yaml 과 같은 순서. 이것이 기준이다
CLASSES = ['scissors', 'rock', 'paper']

# 폴더 이름에 쓰인 말 -> 클래스 번호
# 팀원마다 폴더 이름을 영어로도 한글로도 쓴다. 둘 다 받는다.
WORD2ID = {
    'scissors': 0, 'scissor': 0, 'sci': 0, '가위': 0,
    'rock': 1, '주먹': 1, '바위': 1,
    'paper': 2, '보': 2, '보자기': 2,
}


def expected_from_dirname(name):
    """폴더 이름에서 정답을 읽는다. 해석 못 하면 None.

        'paper_rock'  -> [2, 1]   손 2개 (순서는 상관없다)
        'paper'       -> [2]      손 1개
    """
    ids = []
    for w in name.lower().split('_'):
        if w not in WORD2ID:
            return None
        ids.append(WORD2ID[w])
    return ids if 1 <= len(ids) <= 2 else None


def overlap_ratio(a, b):
    """두 박스가 겹친 넓이가, 작은 쪽 박스의 몇 %인지. 0이면 안 겹친다.

    각 박스는 (cx, cy, w, h) 형태(0~1)다.
    박스가 겹치는 것 자체는 정상이다 — 손이 가까이 붙으면 축에 나란한 네모는
    겹칠 수밖에 없다. 강사님 데이터에서도 손 2개짜리 28장 중 4장이 겹친다.
    다만 강사님 쪽은 최대 5%였다. 그보다 훨씬 크면 '박스를 헐겁게 쳤나'를 의심한다.
    """
    def corners(t):
        cx, cy, w, h = t
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    x1, y1, x2, y2 = corners(a)
    X1, Y1, X2, Y2 = corners(b)
    iw = max(0.0, min(x2, X2) - max(x1, X1))
    ih = max(0.0, min(y2, Y2) - max(y1, Y1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    small = min((x2 - x1) * (y2 - y1), (X2 - X1) * (Y2 - Y1))
    return inter / small * 100.0 if small > 0 else 0.0


# 이 값을 넘으면 '헐거운 박스 의심'으로 알려 준다 (강사님 데이터 최대는 5%)
OVERLAP_WARN = 20.0


def check_classes_txt(d):
    """classes.txt 가 있으면 순서가 맞는지 본다. 이게 틀리면 전부 틀린다."""
    p = os.path.join(d, 'classes.txt')
    if not os.path.exists(p):
        return None, 'classes.txt 없음 (아직 한 장도 저장 안 했거나 PascalVOC로 저장 중)'
    got = [l.strip() for l in open(p, encoding='utf-8') if l.strip()]
    if got == CLASSES:
        return True, 'classes.txt 순서 정상 (%s)' % ' / '.join(got)
    if got[:len(CLASSES)] == CLASSES:
        # 앞 3줄은 맞고 뒤에 더 붙은 경우 = 라벨 이름을 오타로 친 것.
        # 이미 친 박스의 번호는 안 밀리므로 그 줄만 지우면 된다.
        extra = ', '.join(got[len(CLASSES):])
        return False, ('classes.txt 에 없어야 할 클래스가 붙었다 : %s\n'
                       '      -> 라벨 이름을 오타로 친 것이다.\n'
                       '         앞 3줄은 맞으므로 이미 친 박스의 번호는 안 밀린다.\n'
                       '         LabelImg 를 끄고 classes.txt 에서 그 줄만 지우면 된다.\n'
                       '         그 오타 이름으로 친 박스가 있다면 아래에서\n'
                       '         "클래스 번호" 오류로 잡힌다.' % extra)
    return False, ('classes.txt 순서가 다르다!\n'
                   '      지금  : %s\n'
                   '      맞는것: %s\n'
                   '      -> 이 폴더의 번호는 전부 밀려 있다. 라벨을 다시 쳐야 한다.' %
                   (' / '.join(got), ' / '.join(CLASSES)))


def check_dir(d):
    name = os.path.basename(d.rstrip(os.sep))
    exp = expected_from_dirname(name)

    files = os.listdir(d)
    jpgs = sorted(f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    txts = sorted(f for f in files
                  if f.lower().endswith('.txt') and f != 'classes.txt')
    # 사진 없이 라벨만 받은 폴더(나눠 받은 완성본)도 검사할 수 있어야 한다
    bases = sorted({os.path.splitext(f)[0] for f in jpgs} |
                   {os.path.splitext(f)[0] for f in txts})

    print()
    print('=' * 68)
    print(' %s   사진 %d장 / 라벨 %d개' % (name, len(jpgs), len(txts)))
    if exp is None:
        print("  폴더 이름으로 정답을 알 수 없어 '내용 검사'만 한다")
    else:
        print('  폴더 이름이 말하는 정답 : %s  (박스 %d개)'
              % (' + '.join(CLASSES[i] for i in exp), len(exp)))
    print('-' * 68)

    ok_cls, msg = check_classes_txt(d)
    print('  %s' % msg)
    problems = []
    if ok_cls is False:
        # 순서가 어긋나면 이 폴더의 번호가 전부 밀린다. 가장 위험한 사고다
        problems.append('classes.txt 가 기준과 다르다 (위 메시지 참고)')

    done = 0
    nbox = 0
    n_bg = 0            # 빈 txt = 일부러 표시한 배경 사진 (손이 없는 사진)
    loose = []          # 박스가 서로 많이 겹치는 것 (오류는 아니고 확인 대상)
    cls_count = collections.Counter()
    for base in bases:
        p = os.path.join(d, base + '.txt')
        if not os.path.exists(p):
            continue                      # 아직 안 한 것. 뒤에서 세어 알려 준다
        done += 1
        lines = [l.split() for l in open(p) if l.strip()]
        if not lines:
            # 빈 txt = "이 사진은 확인했고, 손이 없다"는 뜻.
            # YOLO 는 이런 사진을 background 로 배운다 ("여기엔 아무것도 없다").
            # 아직 라벨 안 한 사진과 구별하려고 일부러 빈 파일을 만들어 둔 것이다.
            n_bg += 1
            continue
        got = []
        rects = []
        for ln in lines:
            if len(ln) != 5:
                problems.append('%s : 한 줄에 숫자가 5개가 아니다 (%s)' % (base, ' '.join(ln)))
                continue
            cid = int(ln[0])
            if not (0 <= cid < len(CLASSES)):
                problems.append('%s : 클래스 번호 %d (0~2여야 한다)' % (base, cid))
                continue
            vals = [float(v) for v in ln[1:]]
            if any(not (0.0 <= v <= 1.0) for v in vals):
                problems.append('%s : 좌표가 0~1 밖 (%s)' % (base, vals))
            if vals[2] <= 0 or vals[3] <= 0:
                problems.append('%s : 박스 크기가 0이다' % base)
            got.append(cid)
            rects.append(tuple(vals))
            cls_count[CLASSES[cid]] += 1
            nbox += 1

        if len(rects) == 2:
            r = overlap_ratio(rects[0], rects[1])
            if r > OVERLAP_WARN:
                loose.append((base, r))

        if exp is not None:
            if len(got) != len(exp):
                problems.append('%s : 박스가 %d개다 (%d개여야 한다)'
                                % (base, len(got), len(exp)))
            elif sorted(got) != sorted(exp):
                problems.append(
                    '%s : 클래스가 다르다. 적힌 것 [%s] / 폴더가 말하는 것 [%s]'
                    % (base,
                       ', '.join(CLASSES[i] for i in sorted(got)),
                       ', '.join(CLASSES[i] for i in sorted(exp))))

    total = len(bases)
    print('  라벨링 진행 : %d / %d 장   (박스 %d개)' % (done, total, nbox))
    # 사진이 없는 라벨 = 파일 이름이 깨진 것. 조용히 지나가면 그 사진이 통째로 빠진다
    orphan = ({os.path.splitext(f)[0] for f in txts} -
              {os.path.splitext(f)[0] for f in jpgs}) if jpgs else set()
    if orphan:
        print()
        print('  !! 사진이 없는 라벨 %d개 — 파일 이름이 깨졌다' % len(orphan))
        for o in sorted(orphan)[:8]:
            print('     - %s.txt  (%s.jpg 가 없다)' % (o, o))
        print('     -> 이름 끝에 ] 같은 게 붙은 것이라면 그 글자만 지우면 된다.')
        print('        그대로 두면 그 사진은 라벨 없는 사진(배경)으로 학습된다.')
        problems.append('사진이 없는 라벨 %d개 (파일 이름 깨짐)' % len(orphan))

    if n_bg:
        print('  배경으로 표시한 사진 : %d장 (빈 txt = 손이 없는 사진)' % n_bg)
    if jpgs and done < len(jpgs):
        print('  아직 라벨 없는 사진 : %d장' % (len(jpgs) - done))
        print('     -> 손이 없는 사진이라면 "안 하고 두기"가 아니라')
        print('        빈 txt 를 만들어 배경으로 표시하는 편이 낫다:')
        print('        python mark_background.py <사진경로>')
    print('  클래스별 박스 : %s' % dict(cls_count))

    if loose:
        print()
        print('  (참고) 두 박스가 %.0f%% 넘게 겹치는 사진 %d장 — 박스가 헐거운지 확인해 볼 것'
              % (OVERLAP_WARN, len(loose)))
        print('         박스는 그 손의 맨 위/아래/왼쪽/오른쪽 끝에만 맞춘다.')
        print('         다른 손이 박스 안에 들어오는 것은 괜찮지만,')
        print('         다른 손 때문에 박스를 넓히면 안 된다.')
        for b, r in sorted(loose, key=lambda t: -t[1])[:8]:
            print('         - %s : 작은 박스의 %.0f%%가 상대 박스 안' % (b, r))

    if problems:
        print()
        print('  !! 문제 %d건' % len(problems))
        for p_ in problems[:15]:
            print('     - %s' % p_)
        if len(problems) > 15:
            print('     ... 외 %d건' % (len(problems) - 15))
        # 거의 전부 틀리면 순서 사고일 가능성이 높다
        if done and len(problems) > done * 0.8:
            print()
            print('  >> 거의 모든 파일이 틀렸다. 한 장씩 잘못 친 게 아니라')
            print('     classes.txt 순서가 어긋났을 가능성이 높다. 위의 classes.txt 줄을 확인할 것.')
    else:
        print('  문제 없음')

    return len(problems), done, total


if __name__ == '__main__':
    targets = sys.argv[1:]
    if not targets:
        if not os.path.isdir(ROOT):
            raise SystemExit('폴더를 못 찾았다: %s\n'
                             '  경로를 인자로 넘겨라: python check_labels.py <폴더>' % ROOT)
        targets = [os.path.join(ROOT, d) for d in sorted(os.listdir(ROOT))
                   if os.path.isdir(os.path.join(ROOT, d))]

    tot_p = tot_d = tot_n = 0
    for d in targets:
        p, done, n = check_dir(d)
        tot_p += p
        tot_d += done
        tot_n += n

    print()
    print('=' * 68)
    print(' 전체 : 라벨링 %d / %d 장,  문제 %d건' % (tot_d, tot_n, tot_p))
    print('=' * 68)
    if tot_p == 0 and tot_d == tot_n and tot_n:
        print(' 다 끝났고 문제도 없다. make_dataset.py 로 넘어가면 된다.')
