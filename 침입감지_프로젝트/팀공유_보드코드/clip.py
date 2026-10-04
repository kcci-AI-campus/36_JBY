# -*- coding: utf-8 -*-
"""
경보 순간의 앞뒤를 동영상으로 만들어 보호자에게 보내는 모듈.

왜 필요한가
    사진 한 장은 "이미 가까이 갔다"만 보여 준다.
    보호자가 알고 싶은 것은 **어떻게 접근했는가** 다 — 기어서 왔는지, 손을 뻗었는지.
    그러려면 경보가 뜨기 **전**이 담겨야 하는데, 경보를 보고 나서 녹화를 시작하면 이미 늦다.

    그래서 최근 몇 초를 항상 메모리에 돌려 담아 둔다(링 버퍼).
    경보가 뜨면 그 안에 이미 들어 있는 '과거'와, 그 뒤 몇 초를 붙여 한 편으로 만든다.

        ...버리고  [-3초 ────── 경보! ────── +2초]  녹화 끝 -> 전송
                    └─ 링 버퍼에 이미 있던 것 ─┘└ 이제부터 모음 ┘

메모리
    640x480 한 장이 약 0.9 MB. 기본값(앞 3초, 20 FPS)이면 약 55 MB 를 쓴다.
    라즈베리파이 5 는 RAM 이 넉넉해 괜찮지만, 줄이려면 PRE_SEC 이나 SCALE 을 낮춘다.

CPU
    담는 것은 메모리 복사뿐이라 프레임당 0.2 ms 수준이다.
    무거운 일(동영상으로 묶기)은 **경보가 났을 때만**, 그것도 **별도 스레드**에서 한다.

영상에도 박스를 그린다
    보고서와 시연에서는 "모델이 무엇을 보고 경보했는가"가 보여야 설득이 된다.
    날것 화면이 필요하면 detect_baby.py 를 --clip-raw 로 실행한다.

단독 확인 (보드에서 코덱이 되는지 먼저 보라)
    python3 clip.py --check
"""
import os
import sys
import time
import threading
from collections import deque

import numpy as np
import cv2

# 윈도우 콘솔에서도 한글·표 문자가 깨지지 않게 (리눅스에서는 영향 없음)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# ─────────────────────────────────────────────────────────────
def pick_writer(path_noext, size, fps):
    """이 보드에서 실제로 되는 코덱을 찾아 VideoWriter 를 돌려준다.

    mp4 가 휴대폰에서 바로 재생되므로 1순위.
    OpenCV 빌드에 따라 mp4 가 없을 수 있어 avi 로 물러선다.
    """
    trials = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]
    for cc, ext in trials:
        path = path_noext + ext
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w, path, cc
        w.release()
        try:
            os.remove(path)
        except OSError:
            pass
    return None, None, None


class ClipRecorder:
    """최근 프레임을 돌려 담아 두었다가, 경보가 나면 앞뒤를 묶어 보낸다."""

    def __init__(self, notifier=None, pre_sec=3.0, post_sec=2.0,
                 fps=20.0, scale=1.0, outdir="clips", keep_files=True,
                 verbose=True, draw=True, names=None, colors=None,
                 min_gap_sec=30.0, click=None):
        self.noti = notifier
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.fps = float(fps)
        self.scale = float(scale)
        self.outdir = outdir
        self.keep = keep_files
        self.verbose = verbose
        self.draw = draw                 # 영상에도 박스를 그릴 것인가
        self.names = names or {}
        self.colors = colors or {}
        self.min_gap = float(min_gap_sec)   # 영상끼리의 최소 간격
        self.click = click                  # 알림을 눌렀을 때 열 주소
        self._last_trig = 0.0
        self.n_skip = 0

        self.buf = deque(maxlen=max(1, int(self.pre_sec * self.fps)))
        self.recording = False          # 경보 뒤 구간을 모으는 중인가
        self._post = []
        self._t_end = 0.0
        self._text = ""
        self.n_clip = 0
        self.n_fail = 0
        self._busy = False              # 이전 클립을 아직 만드는 중이면 새로 안 받는다
        self._lock = threading.Lock()

        if keep_files:
            os.makedirs(outdir, exist_ok=True)
        if verbose:
            mb = (self.buf.maxlen * 640 * 480 * 3 * self.scale ** 2) / 1048576.0
            print("[클립] 앞 %.0f초 + 뒤 %.0f초, 버퍼 %d장 (약 %.0f MB)"
                  % (self.pre_sec, self.post_sec, self.buf.maxlen, mb))

    # ---------- 루프에서 매 프레임 ----------
    def _annotate(self, f, dets, danger):
        """영상에도 박스를 그린다.

        원본이 아니라 이미 만들어 둔 사본에 그리므로 추론에는 영향이 없다.
        보고서와 시연에서는 '모델이 무엇을 보고 경보했는가'가 보여야 하므로
        날것 영상보다 박스가 그려진 쪽이 쓸모 있다.
        """
        for d in dets or []:
            x1, y1, x2, y2, sc, c = d
            col = self.colors.get(c, (255, 255, 255))
            cv2.rectangle(f, (x1, y1), (x2, y2), col, 2)
            cv2.putText(f, "%s %d%%" % (self.names.get(c, str(c)), int(sc * 100)),
                        (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_PLAIN, 1, col, 2)
        if danger:
            cv2.rectangle(f, (0, 0), (f.shape[1] - 1, f.shape[0] - 1), (0, 0, 255), 6)
            cv2.putText(f, "DANGER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
        return f

    def push(self, frame, dets=None, danger=False):
        """프레임을 담는다. 원본을 건드리지 않도록 반드시 사본을 넣는다."""
        if self.scale != 1.0:
            f = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        else:
            f = frame.copy()
        if self.draw:
            self._annotate(f, dets, danger)
        if self.recording:
            self._post.append(f)
            if time.time() >= self._t_end:
                self._finish()
        else:
            self.buf.append(f)

    def trigger(self, text=""):
        """경보 발생. 지금부터 post_sec 만큼 더 모은 뒤 만들어 보낸다."""
        if self.recording or self._busy:
            return False
        now = time.time()
        if now - self._last_trig < self.min_gap:
            # 경보는 3초마다 다시 울릴 수 있지만 영상까지 그때마다 보내면
            # 보호자 휴대폰이 영상으로 도배된다. 사진 쪽과 같은 이유로 제한한다.
            self.n_skip += 1
            return False
        self._last_trig = now
        self.recording = True
        self._post = []
        self._t_end = time.time() + self.post_sec
        self._text = text
        return True

    # ---------- 내부 ----------
    def _finish(self):
        frames = list(self.buf) + self._post
        self.buf.clear()
        self._post = []
        self.recording = False
        if not frames:
            return
        self._busy = True
        threading.Thread(target=self._encode_and_send,
                         args=(frames, self._text), daemon=True).start()

    def _encode_and_send(self, frames, text):
        """무거운 일은 여기서. 루프와 따로 돈다."""
        try:
            h, w = frames[0].shape[:2]
            stamp = time.strftime("%m%d_%H%M%S")
            base = os.path.join(self.outdir if self.keep else ".", "clip_" + stamp)
            wr, path, cc = pick_writer(base, (w, h), self.fps)
            if wr is None:
                self.n_fail += 1
                if self.verbose:
                    print("[클립] 만들 수 있는 코덱이 없습니다. python3 clip.py --check 로 확인하세요")
                return
            for f in frames:
                wr.write(f)
            wr.release()

            mb = os.path.getsize(path) / 1048576.0
            self.n_clip += 1
            if self.verbose:
                print("[클립] %s  %d장 %.1f초 %.2f MB (%s)"
                      % (os.path.basename(path), len(frames),
                         len(frames) / self.fps, mb, cc))

            if self.noti:
                with open(path, "rb") as f:
                    data = f.read()
                self.noti.send_file(text, data, os.path.basename(path),
                                    "video/mp4" if path.endswith(".mp4") else "video/x-msvideo",
                                    title="⚠️ 위험 감지 (영상)", click=self.click)
            if not self.keep:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except Exception as e:
            self.n_fail += 1
            if self.verbose:
                print("[클립] 실패 :", e)
        finally:
            self._busy = False

    def summary(self):
        return ("  클립          : 만듦 %d / 실패 %d / 건너뜀 %d"
                % (self.n_clip, self.n_fail, self.n_skip))


# ─────────────────────────────────────────────────────────────
def do_check():
    """이 보드에서 어떤 코덱으로 동영상을 만들 수 있는지 확인한다."""
    print("=" * 60)
    print(" 이 보드에서 쓸 수 있는 동영상 코덱 확인")
    print("=" * 60)
    size, fps = (320, 240), 20.0
    img = np.full((240, 320, 3), 60, np.uint8)
    ok_any = False
    for cc, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi")]:
        path = "_codec_test" + ext
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            for i in range(10):
                cv2.putText(img, str(i), (140, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 200, 255), 2)
                w.write(img)
            w.release()
            sz = os.path.getsize(path) if os.path.exists(path) else 0
            print("  %-6s %-5s  됨    (%d bytes)" % (cc, ext, sz))
            ok_any = ok_any or sz > 0
        else:
            print("  %-6s %-5s  안 됨" % (cc, ext))
            w.release()
        try:
            os.remove(path)
        except OSError:
            pass
    print("-" * 60)
    if ok_any:
        print("  쓸 수 있는 코덱이 있습니다. 클립 기능을 켜도 됩니다.")
        print("      python3 detect_baby.py best_int8.tflite --notify --clip")
    else:
        print("  동영상을 만들 수 없습니다. 아래를 설치해 보세요 :")
        print("      sudo apt install -y ffmpeg")
    print("=" * 60)
    return 0 if ok_any else 1


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(do_check())
    print(__doc__)
    print("사용법 :  python3 clip.py --check    코덱 확인")
