# -*- coding: utf-8 -*-
"""
팀원들이 각자 만든 데이터셋 폴더를 하나로 합친다.

입력 구조 (팀 드라이브에서 받은 그대로)
    dataset/
        baby/    images/{train,test}   labels/{train,test}
        baby3/   ...
        knife/   ...
        Outlet/  ...
        Baby2/   ...

출력 구조 (Ultralytics YOLO 학습용)
    merged/
        images/train  images/val
        labels/train  labels/val
        data.yaml

하는 일
  1) 파일 이름 앞에 출처 폴더명을 붙인다  ->  이름 충돌 방지
     (여러 사람이 같은 촬영 스크립트를 써서 파일명이 겹친다)
  2) 라벨이 없는 사진은 건너뛴다 (실수로 빠진 것일 수 있으므로 목록으로 알려준다)
  3) test -> val 로 옮긴다 (Ultralytics는 train/val 이름을 쓴다)
  4) 클래스 번호를 검사하고, 클래스별 박스 수를 집계한다
  5) data.yaml 을 만든다

사용법
    python merge_dataset.py <dataset 폴더> [출력폴더]
"""
import os
import sys
import shutil
import collections

CLASSES = ["baby", "adult", "knife", "outlet"]
IMG_EXT = (".jpg", ".jpeg", ".png")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return
    src = args[0]
    dst = args[1] if len(args) > 1 else os.path.join(os.path.dirname(os.path.abspath(src)), "merged")

    if not os.path.isdir(src):
        print("폴더가 없습니다:", src)
        return

    for sp in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(dst, kind, sp), exist_ok=True)

    box = collections.Counter()
    per_src = collections.defaultdict(collections.Counter)
    n_img = collections.Counter()
    skipped = []
    bad = []

    # 합친 결과물 폴더가 원본과 같은 자리에 있으면 같은 데이터를 두 번 넣게 된다.
    # (팀 드라이브의 dataset/ 안에 merged, merged_dataset 가 같이 있다)
    SKIP = ("merged", "merged_dataset", "out", "result")

    for ds in sorted(os.listdir(src)):
        p = os.path.join(src, ds)
        if not os.path.isdir(p):
            continue
        if ds.lower().startswith("merged") or ds.lower() in SKIP or ds.startswith("_"):
            print("건너뜀 (합친 결과물) :", ds)
            continue
        for sp_in, sp_out in (("train", "train"), ("test", "val"), ("val", "val")):
            d_img = os.path.join(p, "images", sp_in)
            d_lab = os.path.join(p, "labels", sp_in)
            if not os.path.isdir(d_img):
                continue
            for f in sorted(os.listdir(d_img)):
                if not f.lower().endswith(IMG_EXT):
                    continue
                stem = os.path.splitext(f)[0]
                lab = os.path.join(d_lab, stem + ".txt")
                if not os.path.exists(lab):
                    skipped.append("%s/%s/%s" % (ds, sp_in, f))
                    continue

                new = "%s__%s" % (ds, stem)          # 출처를 이름에 남긴다
                shutil.copy2(os.path.join(d_img, f),
                             os.path.join(dst, "images", sp_out,
                                          new + os.path.splitext(f)[1]))

                lines = []
                for i, line in enumerate(open(lab, encoding="utf-8", errors="replace"), 1):
                    line = line.strip()
                    if not line:
                        continue
                    q = line.split()
                    if len(q) != 5:
                        bad.append("%s:%d 항목 %d개" % (lab, i, len(q)))
                        continue
                    try:
                        ci = int(q[0])
                        v = [float(x) for x in q[1:]]
                    except ValueError:
                        bad.append("%s:%d 숫자 아님" % (lab, i))
                        continue
                    if not (0 <= ci < len(CLASSES)):
                        bad.append("%s:%d 클래스 번호 %d" % (lab, i, ci))
                        continue
                    if not all(0.0 <= x <= 1.0 for x in v):
                        bad.append("%s:%d 좌표 범위 밖" % (lab, i))
                        continue
                    lines.append(line)
                    box[ci] += 1
                    per_src[ds][ci] += 1

                with open(os.path.join(dst, "labels", sp_out, new + ".txt"),
                          "w", encoding="utf-8") as fo:
                    fo.write("\n".join(lines) + ("\n" if lines else ""))
                n_img[sp_out] += 1

    # data.yaml
    yaml = os.path.join(dst, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % os.path.abspath(dst).replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("nc: %d\n" % len(CLASSES))
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))

    # 결과
    print("=" * 66)
    print(" 출력 : %s" % os.path.abspath(dst))
    print("-" * 66)
    print(" train %4d장 / val %4d장 / 합계 %4d장"
          % (n_img["train"], n_img["val"], sum(n_img.values())))
    print("-" * 66)
    print(" 클래스별 박스")
    for i, c in enumerate(CLASSES):
        mark = "   <-- 데이터 없음" if not box.get(i) else ""
        print("   %d %-8s %5d개%s" % (i, c, box.get(i, 0), mark))
    print("-" * 66)
    print(" 출처별")
    for ds in sorted(per_src):
        s = ", ".join("%s %d" % (CLASSES[k], v) for k, v in sorted(per_src[ds].items()))
        print("   %-8s %s" % (ds, s))
    print("-" * 66)
    if skipped:
        print(" 라벨이 없어 제외한 사진 %d장" % len(skipped))
        for s in skipped[:10]:
            print("     ", s)
        if len(skipped) > 10:
            print("      ... 외 %d장" % (len(skipped) - 10))
    if bad:
        print(" 라벨 오류 %d건" % len(bad))
        for b in bad[:10]:
            print("     ", b)
    if not skipped and not bad:
        print(" 문제 없음")
    print("=" * 66)
    print(" data.yaml : %s" % yaml)


if __name__ == "__main__":
    main()
