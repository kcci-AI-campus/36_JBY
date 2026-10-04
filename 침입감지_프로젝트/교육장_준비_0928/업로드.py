# -*- coding: utf-8 -*-
"""보드로_올릴것/ 폴더를 보드의 ~/work/baby/anseong/ 에 올린다 (없으면 만든다).

사용 (이 폴더에서):
    python 업로드.py 10.10.15.61 willtek 비밀번호            # 교육장 보드
    python 업로드.py 192.168.123.104 willtek 1234           # 지인 보드
    python 업로드.py 10.10.15.61 willtek 비밀번호 --only 코드   # .py .sh 만 (모델·영상 제외, 몇 초)

필요: pip install paramiko  (이 노트북 파이썬 3.10 에는 설치돼 있음)
"""
import os, sys, time
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if len(sys.argv) < 4:
    print(__doc__); sys.exit(1)
host, user, pw = sys.argv[1:4]
only_code = "--only" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "보드로_올릴것")
DST = "work/baby/anseong"

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("접속 중:", user + "@" + host)
c.connect(host, username=user, password=pw, timeout=10)

if "--get" in sys.argv:
    # 결과 내려받기: 보드의 *_summary.csv, *_log.csv, 요약 출력을 결과_0928/ 로
    out_dir = os.path.join(HERE, "결과_0928"); os.makedirs(out_dir, exist_ok=True)
    sftp = c.open_sftp(); home = sftp.normalize(".")
    rdir = "%s/%s" % (home, DST)
    names = [n for n in sftp.listdir(rdir) if n.endswith(("_summary.csv", "_log.csv", ".txt")) and not n.startswith(("s4_", "s3_", "lp_"))]
    for n in sorted(names):
        sftp.get("%s/%s" % (rdir, n), os.path.join(out_dir, n)); print("  받음:", n)
    sftp.close(); c.close()
    print("완료: %d개 → %s" % (len(names), out_dir)); sys.exit(0)
c.exec_command("mkdir -p ~/%s" % DST)[1].read()
sftp = c.open_sftp()
home = sftp.normalize(".")
files = sorted(os.listdir(SRC))
if only_code:
    files = [f for f in files if f.endswith((".py", ".sh", ".json"))]
t0 = time.time(); total = 0
for f in files:
    lp = os.path.join(SRC, f); rp = "%s/%s/%s" % (home, DST, f)
    sz = os.path.getsize(lp)
    try:
        if sftp.stat(rp).st_size == sz:
            print("  같음, 건너뜀:", f); continue
    except IOError:
        pass
    sftp.put(lp, rp); total += sz
    print("  올림: %-32s %6.1f MB" % (f, sz / 1e6))
sftp.close()
print("완료: %.1f MB, %.0f초" % (total / 1e6, time.time() - t0))
out = c.exec_command("cd ~/%s && sed -i 's/\\r$//' *.sh *.py && chmod +x *.sh && source ~/work/env/bin/activate && python -c \"import ai_edge_litert, cv2; print('실행 환경 OK: LiteRT', ai_edge_litert.__version__, 'OpenCV', cv2.__version__)\" 2>&1 | tail -1" % DST)[1].read().decode("utf-8", "replace")
print(out.strip())
print("다음: 보드 터미널에서  cd ~/work/baby/anseong  →  bash 실험.sh 목록")
c.close()
