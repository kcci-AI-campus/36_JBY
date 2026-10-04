# -*- coding: utf-8 -*-
"""
라벨링 끝난 폴더를 YOLO 표준 구조(train / test)로 나눈다.

    <출력폴더>/
        images/train/*.jpg   images/test/*.jpg
        labels/train/*.txt   labels/test/*.txt
        classes.txt

왜 '무작위'로 안 나누는가
------------------------
자동 촬영은 0.8초마다 찍으므로 **앞뒤 사진이 거의 같은 그림**이다.
무작위로 섞어 나누면 거의 같은 사진 두 장이 train과 test에 하나씩 들어가서,
모델이 이미 본 것을 시험 보는 셈이 된다. 점수가 실제보다 높게 나오고,
그 점수를 믿을 수 없게 되는 것이 진짜 손해다.

그래서 **파일 이름 순으로 늘어놓고 덩어리째** 나눈다.
(촬영 순서 = 시간 순서이므로, 덩어리로 자르면 비슷한 장면이 한쪽에만 모인다)

사용법
    python split_dataset.py <라벨링한 폴더> [출력폴더] [--ratio 0.2] [--blocks 10]

    python split_dataset.py captures\pro outlet_dataset
"""
import os
import sys
import shutil

CLASSES = ["baby", "adult", "knife", "outlet"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return
    src = args[0]
    dst = args[1] if len(args) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(src)) or ".", "dataset_out")

    ratio = 0.2      # test 비율
    blocks = 10      # 몇 덩어리로 잘라서 나눌지
    for i, a in enumerate(sys.argv):
        if a == "--ratio" and i + 1 < len(sys.argv):
            ratio = float(sys.argv[i + 1])
        if a == "--blocks" and i + 1 < len(sys.argv):
            blocks = int(sys.argv[i + 1])

    if not os.path.isdir(src):
        print("폴더가 없습니다:", src)
        return

    pairs = []
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(".jpg"):
            continue
        stem = os.path.splitext(f)[0]
        t = os.path.join(src, stem + ".txt")
        if os.path.exists(t):
            pairs.append((f, stem + ".txt"))

    if not pairs:
        print("사진+라벨 짝이 없습니다. 먼저 cleanup_labels.py 를 돌리세요.")
        return

    # ── 덩어리째 나누기 ───────────────────────────────────
    n = len(pairs)
    size = max(1, n // blocks)
    chunks = [pairs[i:i + size] for i in range(0, n, size)]
    n_test_chunk = max(1, int(round(len(chunks) * ratio)))
    # 앞·중간·뒤에서 고르게 뽑는다 (한쪽에 몰리지 않게)
    step = max(1, len(chunks) // n_test_chunk)
    test_idx = set(range(0, len(chunks), step))
    while len(test_idx) > n_test_chunk:
        test_idx.pop()

    train, test = [], []
    for i, ch in enumerate(chunks):
        (test if i in test_idx else train).extend(ch)

    # ── 폴더 만들고 복사 ──────────────────────────────────
    for sub in ("images/train", "images/test", "labels/train", "labels/test"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)

    for split, items in (("train", train), ("test", test)):
        for jpg, txt in items:
            shutil.copy2(os.path.join(src, jpg),
                         os.path.join(dst, "images", split, jpg))
            shutil.copy2(os.path.join(src, txt),
                         os.path.join(dst, "labels", split, txt))

    with open(os.path.join(dst, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASSES) + "\n")

    # ── 결과 ──────────────────────────────────────────────
    def count_boxes(split, items):
        c = {}
        for _, txt in items:
            for line in open(os.path.join(dst, "labels", split, txt), encoding="utf-8"):
                if line.strip():
                    ci = int(line.split()[0])
                    name = CLASSES[ci] if 0 <= ci < len(CLASSES) else "??%d" % ci
                    c[name] = c.get(name, 0) + 1
        return c

    print("=" * 56)
    print(" 원본 : %s  (짝 맞는 사진 %d장)" % (src, n))
    print(" 출력 : %s" % os.path.abspath(dst))
    print("-" * 56)
    print(" train %4d장   박스 %s" % (len(train), count_boxes("train", train)))
    print(" test  %4d장   박스 %s" % (len(test), count_boxes("test", test)))
    print("-" * 56)
    print(" 구조")
    print("   images/train  images/test")
    print("   labels/train  labels/test")
    print("   classes.txt")
    print("=" * 56)


if __name__ == "__main__":
    main()
