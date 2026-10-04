# -*- coding: utf-8 -*-
"""
라벨링 정리·검사 도구

라벨링을 끝낸 뒤(또는 중간에) 한 번 돌리면 이런 것들을 정리·점검한다.
  1) 빈 라벨 파일   -> 사진과 함께 _skipped 로 옮긴다
                      (대상이 있는데 빈 라벨이면 "없다"고 잘못 가르치기 때문)
  2) 짝 없는 라벨   -> _orphan_labels 로 옮긴다 (사진이 사라진 라벨)
  3) classes.txt    -> 4줄(baby/adult/knife/outlet)로 맞춘다
                      (LabelImg가 제멋대로 덮어쓰는 일이 있다)
  4) 검사           -> 클래스 번호 분포, 좌표 범위, 박스 크기 이상

사용법
    python cleanup_labels.py                 # 기본 폴더(captures 아래 전부)
    python cleanup_labels.py 폴더경로         # 특정 폴더만
    python cleanup_labels.py 폴더경로 --check # 옮기지 않고 검사만
"""
import os
import sys
import shutil
import collections

CLASSES = ["baby", "adult", "knife", "outlet"]
DEFAULT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")


def tidy(folder, check_only=False):
    print("=" * 62)
    print(" 폴더 :", folder)
    if not os.path.isdir(folder):
        print("  폴더가 없습니다.")
        return

    files = os.listdir(folder)
    jpgs = [f for f in files if f.lower().endswith(".jpg")]
    txts = [f for f in files if f.lower().endswith(".txt") and f != "classes.txt"]
    if not jpgs and not txts:
        print("  사진도 라벨도 없습니다.")
        return

    skip_dir = os.path.join(folder, "_skipped")
    orph_dir = os.path.join(folder, "_orphan_labels")

    # ── 1) classes.txt 맞추기 ──────────────────────────────
    cp = os.path.join(folder, "classes.txt")
    cur = []
    if os.path.exists(cp):
        cur = [l.strip() for l in open(cp, encoding="utf-8") if l.strip()]
    if cur != CLASSES:
        print("  [classes.txt] %s -> %s%s"
              % (cur if cur else "(없음/빈 파일)", CLASSES,
                 "  (검사만)" if check_only else ""))
        if not check_only:
            open(cp, "w", encoding="utf-8").write("\n".join(CLASSES) + "\n")
    else:
        print("  [classes.txt] 정상")

    # ── 2) 빈 라벨 / 짝 없는 라벨 ──────────────────────────
    stems_j = {os.path.splitext(f)[0] for f in jpgs}
    empty, orphan = [], []
    for t in txts:
        p = os.path.join(folder, t)
        stem = os.path.splitext(t)[0]
        body = open(p, encoding="utf-8").read().strip()
        if stem not in stems_j:
            orphan.append(t)
        elif not body:
            empty.append(t)

    def move(names, dest, why):
        if not names:
            return
        print("  [%s] %d개%s" % (why, len(names), "  (검사만)" if check_only else ""))
        for n in names[:8]:
            print("      ", n)
        if len(names) > 8:
            print("       ... 외 %d개" % (len(names) - 8))
        if check_only:
            return
        os.makedirs(dest, exist_ok=True)
        for n in names:
            shutil.move(os.path.join(folder, n), os.path.join(dest, n))
            jp = os.path.join(folder, os.path.splitext(n)[0] + ".jpg")
            if os.path.exists(jp):
                shutil.move(jp, os.path.join(dest, os.path.basename(jp)))

    move(empty, skip_dir, "빈 라벨 -> 사진과 함께 _skipped 로")
    move(orphan, orph_dir, "짝 없는 라벨 -> _orphan_labels 로")

    # ── 2-b) --finish : 라벨이 아예 없는 사진도 뺀다 ───────
    #   라벨링 '중'에는 아직 안 한 사진일 수 있으므로 기본 동작이 아니다.
    #   전부 끝낸 뒤에만 --finish 를 붙여서 쓴다.
    if "--finish" in sys.argv:
        stems_t = {os.path.splitext(f)[0] for f in os.listdir(folder)
                   if f.lower().endswith(".txt") and f != "classes.txt"}
        todo = sorted(f for f in os.listdir(folder)
                      if f.lower().endswith(".jpg")
                      and os.path.splitext(f)[0] not in stems_t)
        if todo:
            print("  [라벨 없는 사진 -> _skipped 로] %d장%s"
                  % (len(todo), "  (검사만)" if check_only else ""))
            for n_ in todo[:8]:
                print("      ", n_)
            if len(todo) > 8:
                print("       ... 외 %d장" % (len(todo) - 8))
            if not check_only:
                os.makedirs(skip_dir, exist_ok=True)
                for n_ in todo:
                    shutil.move(os.path.join(folder, n_),
                                os.path.join(skip_dir, n_))

    # ── 3) 내용 검사 ───────────────────────────────────────
    files = os.listdir(folder)
    jpgs = [f for f in files if f.lower().endswith(".jpg")]
    txts = [f for f in files if f.lower().endswith(".txt") and f != "classes.txt"]

    cnt = collections.Counter()
    bad = []
    tiny = []
    for t in txts:
        for i, line in enumerate(open(os.path.join(folder, t), encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) != 5:
                bad.append("%s:%d 항목이 5개가 아님" % (t, i))
                continue
            try:
                ci = int(p[0])
                x, y, w, h = (float(v) for v in p[1:])
            except ValueError:
                bad.append("%s:%d 숫자가 아님" % (t, i))
                continue
            if not (0 <= ci < len(CLASSES)):
                bad.append("%s:%d 클래스 번호 %d (0~%d 이어야 함)"
                           % (t, i, ci, len(CLASSES) - 1))
                continue
            cnt[CLASSES[ci]] += 1
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                bad.append("%s:%d 좌표가 0~1 밖" % (t, i))
            if w <= 0.005 or h <= 0.005:
                tiny.append("%s:%d 박스가 너무 작음 (%.3f x %.3f)" % (t, i, w, h))

    print("")
    print("  사진 %d장 / 라벨 %d개 / 라벨 안 된 사진 %d장"
          % (len(jpgs), len(txts), len(jpgs) - len(txts)))
    print("  박스 %d개 :" % sum(cnt.values()),
          ", ".join("%s %d" % (k, cnt[k]) for k in CLASSES if cnt[k]) or "없음")
    if bad:
        print("  [오류] %d건" % len(bad))
        for b in bad[:10]:
            print("      ", b)
    if tiny:
        print("  [주의] 아주 작은 박스 %d건 (잘못 클릭한 것일 수 있음)" % len(tiny))
        for b in tiny[:5]:
            print("      ", b)
    if not bad and not tiny:
        print("  검사 통과")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv

    if args:
        targets = [args[0]]
    else:
        base = DEFAULT_BASE
        targets = [os.path.join(base, d) for d in sorted(os.listdir(base))
                   if os.path.isdir(os.path.join(base, d))] if os.path.isdir(base) else []
        if not targets:
            print("captures 아래에 폴더가 없습니다. 경로를 직접 지정하세요.")
            return

    for t in targets:
        if os.path.basename(t).startswith("_"):
            continue
        tidy(t, check_only)
    print("=" * 62)


if __name__ == "__main__":
    main()
