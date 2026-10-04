# -*- coding: utf-8 -*-
"""
손이 없는 사진을 '배경 사진'으로 표시한다.

LabelImg 는 박스가 하나도 없으면 파일을 저장하지 않는다.
그래서 "손이 없는 사진"과 "아직 라벨 안 한 사진"이 구별되지 않는다.
둘 다 txt 가 없기 때문이다.

이 도구는 **내용이 빈 txt** 를 만들어 준다.
    빈 txt  = "확인했고, 여기엔 손이 없다"
    txt 없음 = "아직 안 했다"

YOLO 는 빈 txt 든 txt 없음이든 똑같이 background(아무것도 없음)로 학습한다.
구별은 사람과 검사기를 위한 것이다.

배경 사진은 왜 필요한가:
    얼굴·벽·옷을 "손이 아니다"라고 가르치는 유일한 방법이다.
    강사님 데이터에는 이런 사진이 0장이라, 모델이 얼굴을 손으로 잡는다.
    전체의 10% 정도가 권장치다.

쓰는 법
    python mark_background.py 사진.jpg                 한 장
    python mark_background.py 폴더/img_0001.jpg 폴더/img_0007.jpg
    python mark_background.py --list 폴더              그 폴더에서 라벨 없는 사진 목록만 보기
"""
import os
import sys


def mark(img):
    if not os.path.exists(img):
        print('  없는 파일: %s' % img)
        return False
    txt = os.path.splitext(img)[0] + '.txt'
    if os.path.exists(txt) and os.path.getsize(txt) > 0:
        print('  이미 라벨이 있다(박스 있음), 건너뜀: %s' % os.path.basename(img))
        return False
    open(txt, 'w').close()          # 내용이 빈 파일
    print('  배경으로 표시: %s' % os.path.basename(txt))
    return True


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit

    if args[0] == '--list':
        d = args[1]
        todo = []
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if not os.path.exists(os.path.join(d, os.path.splitext(f)[0] + '.txt')):
                todo.append(f)
        print('라벨이 없는 사진 %d장:' % len(todo))
        for f in todo:
            print('   ', f)
        raise SystemExit

    n = sum(1 for a in args if mark(a))
    print('%d장을 배경으로 표시했다.' % n)
