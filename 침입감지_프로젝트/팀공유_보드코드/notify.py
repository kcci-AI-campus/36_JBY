# -*- coding: utf-8 -*-
"""
위험 상황을 보호자 휴대폰으로 알리는 모듈.

왜 별도 파일인가
    보내는 일은 네트워크를 타므로 느리고, 가끔 실패한다.
    추론 루프 안에서 그냥 호출하면 그동안 영상이 멈춘다(프레임을 놓친다).
    그래서 **별도 스레드**에 맡기고 루프는 즉시 돌아가게 만들었다.
    보내기가 실패해도 프로그램은 절대 죽지 않는다 — 감시가 멈추는 게 더 큰 사고다.

지원하는 방식 두 가지
    ntfy      가입 불필요. 토픽 이름만 정하면 된다. 가장 빠르게 시연 가능
    telegram  봇을 하나 만들어야 하지만, 실제 서비스에 가깝다

설정 파일 notify.json (이 파일과 같은 폴더)
    {
      "backend": "ntfy",
      "ntfy":     { "topic": "baby-guard-team10-a7x2k9" },
      "telegram": { "token": "123456:ABC-DEF...", "chat_id": "987654321" },
      "min_interval_sec": 30,
      "send_photo": true
    }

    ** notify.json 은 절대 GitHub 에 올리지 않는다. ** 텔레그램 토큰은 비밀번호와 같다.
    .gitignore 에 넣어 두었다.

준비
    pip install requests

시험 (추론 없이 보내기만 확인)
    python3 notify.py --test

쓰는 쪽
    from notify import Notifier
    noti = Notifier()                       # notify.json 을 읽는다
    noti.send("아기가 커터칼에 접근", frame)   # 즉시 돌아온다. 보내기는 뒤에서
    ...
    noti.close()                            # 끝낼 때 (남은 것 마저 보내고 정리)
"""
import os
import io
import sys
import json
import time
import queue
import threading

try:
    import requests
except ImportError:
    requests = None

try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_PATH = os.path.join(HERE, "notify.json")

DEFAULT = {
    "backend": "ntfy",
    "ntfy": {"server": "https://ntfy.sh", "topic": ""},
    "telegram": {"token": "", "chat_id": ""},
    "min_interval_sec": 30,
    "send_photo": True,
    "timeout_sec": 10,
}


def load_conf(path=CONF_PATH):
    """설정을 읽는다. 없으면 기본값 + 만드는 법 안내."""
    conf = json.loads(json.dumps(DEFAULT))       # 깊은 복사
    if not os.path.exists(path):
        return conf, False
    try:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
    except Exception as e:
        print("[알림] notify.json 을 읽지 못했습니다 : %s" % e)
        return conf, False
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(conf.get(k), dict):
            conf[k].update(v)
        else:
            conf[k] = v
    return conf, True


# ─────────────────────────────────────────────────────────────
#  보내는 방법 두 가지
# ─────────────────────────────────────────────────────────────
def _send_ntfy(conf, title, text, jpg, timeout, fname=None, mime=None, click=None):
    """ntfy.sh — 가입 없이 토픽 이름만으로 쓰는 푸시 알림.

    jpg 자리에는 사진뿐 아니라 동영상 등 아무 파일의 바이트를 넣을 수 있다.
    fname / mime 을 주면 그 이름·형식으로 첨부된다.
    """
    c = conf["ntfy"]
    topic = c.get("topic", "").strip()
    if not topic:
        return False, "ntfy topic 이 비어 있습니다"
    url = c.get("server", "https://ntfy.sh").rstrip("/") + "/" + topic

    # 한글은 HTTP 헤더에 그냥 못 넣는다. RFC 2047 방식으로 감싼다.
    def hdr(s):
        import base64
        if all(ord(ch) < 128 for ch in s):
            return s
        return "=?UTF-8?B?" + base64.b64encode(s.encode("utf-8")).decode() + "?="

    headers = {"Title": hdr(title), "Priority": "urgent", "Tags": "rotating_light"}
    if click:
        # 알림을 누르면 이 주소가 열린다. 실시간 화면으로 바로 넘어가게 한다.
        headers["Click"] = click

    if jpg is not None:
        # 첨부는 본문에 바이트로 싣고, 설명은 Message 헤더로 보낸다
        headers["Filename"] = fname or "danger.jpg"
        headers["Message"] = hdr(text)
        if mime:
            headers["Content-Type"] = mime
        r = requests.put(url, data=jpg, headers=headers, timeout=timeout)
    else:
        r = requests.post(url, data=text.encode("utf-8"),
                          headers=headers, timeout=timeout)
    ok = 200 <= r.status_code < 300
    return ok, ("HTTP %d" % r.status_code) + ("" if ok else " " + r.text[:200])


def _send_telegram(conf, title, text, jpg, timeout, fname=None, mime=None, click=None):
    """텔레그램 봇 — 사진·동영상과 설명을 한 번에 보낸다."""
    c = conf["telegram"]
    token, chat = c.get("token", "").strip(), str(c.get("chat_id", "")).strip()
    if not token or not chat:
        return False, "telegram token 또는 chat_id 가 비어 있습니다"
    base = "https://api.telegram.org/bot" + token
    caption = ("%s\n%s" % (title, text))[:1024]

    if jpg is not None:
        # 동영상이면 sendVideo, 아니면 sendPhoto
        is_video = bool(mime and mime.startswith("video"))
        api = "/sendVideo" if is_video else "/sendPhoto"
        key = "video" if is_video else "photo"
        r = requests.post(base + api,
                          data={"chat_id": chat, "caption": caption},
                          files={key: (fname or "danger.jpg", jpg,
                                       mime or "image/jpeg")},
                          timeout=timeout)
    else:
        r = requests.post(base + "/sendMessage",
                          data={"chat_id": chat, "text": caption},
                          timeout=timeout)
    ok = False
    try:
        ok = bool(r.json().get("ok"))
    except Exception:
        pass
    return ok, ("HTTP %d" % r.status_code) + ("" if ok else " " + r.text[:200])


SENDERS = {"ntfy": _send_ntfy, "telegram": _send_telegram}


# ─────────────────────────────────────────────────────────────
class Notifier:
    """알림을 별도 스레드에서 보낸다. send() 는 즉시 돌아온다."""

    def __init__(self, conf=None, verbose=True):
        if conf is None:
            conf, found = load_conf()
            if verbose and not found:
                print("[알림] notify.json 이 없습니다. 알림을 보내지 않습니다.")
                print("       만드는 법 : python3 notify.py --init")
        self.conf = conf
        self.verbose = verbose
        self.backend = conf.get("backend", "ntfy")
        self.min_gap = float(conf.get("min_interval_sec", 30))
        self.want_photo = bool(conf.get("send_photo", True))
        self.timeout = float(conf.get("timeout_sec", 10))

        self.enabled = self._check()
        self.n_sent = 0
        self.n_fail = 0
        self.n_skip = 0
        self._last = 0.0
        self._q = queue.Queue(maxsize=8)     # 밀리면 버린다. 쌓아 두면 의미가 없다
        self._done = threading.Event()
        self._th = None
        if self.enabled:
            self._th = threading.Thread(target=self._worker, daemon=True)
            self._th.start()
            if verbose:
                print("[알림] %s 로 보냅니다 (최소 간격 %.0f초, 사진 %s)"
                      % (self.backend, self.min_gap, "포함" if self.want_photo else "없음"))

    def _check(self):
        if requests is None:
            if self.verbose:
                print("[알림] requests 가 없어 알림을 끕니다.  pip install requests")
            return False
        if self.backend not in SENDERS:
            if self.verbose:
                print("[알림] 모르는 backend : %s (ntfy / telegram)" % self.backend)
            return False
        if self.backend == "ntfy" and not self.conf["ntfy"].get("topic", "").strip():
            return False
        if self.backend == "telegram" and not self.conf["telegram"].get("token", "").strip():
            return False
        return True

    # ---------- 쓰는 쪽 ----------
    def send(self, text, frame=None, title="⚠️ 위험 감지", click=None):
        """알림을 예약한다. 네트워크를 기다리지 않고 바로 돌아온다."""
        if not self.enabled:
            return False
        now = time.time()
        if now - self._last < self.min_gap:      # 너무 자주 보내지 않는다
            self.n_skip += 1
            return False
        self._last = now

        jpg = None
        if frame is not None and self.want_photo and cv2 is not None:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                jpg = buf.tobytes()
        try:
            self._q.put_nowait((title, text, jpg, None, None, click))
            return True
        except queue.Full:
            self.n_skip += 1
            return False

    def send_file(self, text, data, fname, mime, title="⚠️ 위험 감지",
                  ignore_interval=True, click=None):
        """동영상 같은 파일을 붙여 보낸다.

        ignore_interval 이 True 면 최소 간격 제한을 건너뛴다.
        클립은 이미 경보 1건당 한 번만 만들어지므로 따로 제한할 필요가 없고,
        방금 보낸 사진 때문에 정작 중요한 영상이 막히면 곤란하기 때문이다.
        """
        if not self.enabled:
            return False
        if not ignore_interval and time.time() - self._last < self.min_gap:
            self.n_skip += 1
            return False
        try:
            self._q.put_nowait((title, text, data, fname, mime, click))
            return True
        except queue.Full:
            self.n_skip += 1
            return False

    def close(self, wait_sec=5.0):
        """남은 것을 마저 보내고 정리한다."""
        if not self.enabled:
            return
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < wait_sec:
            time.sleep(0.1)
        self._done.set()
        if self._th:
            self._th.join(timeout=2.0)

    def summary(self):
        if not self.enabled:
            return "  알림          : 사용 안 함"
        return ("  알림          : 보냄 %d / 실패 %d / 건너뜀 %d (%s)"
                % (self.n_sent, self.n_fail, self.n_skip, self.backend))

    # ---------- 내부 ----------
    def _worker(self):
        fn = SENDERS[self.backend]
        while not self._done.is_set() or not self._q.empty():
            try:
                title, text, jpg, fname, mime, click = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                ok, info = fn(self.conf, title, text, jpg, self.timeout,
                              fname, mime, click)
            except Exception as e:                 # 네트워크가 끊겨도 죽지 않는다
                ok, info = False, str(e)
            if ok:
                self.n_sent += 1
            else:
                self.n_fail += 1
                if self.verbose:
                    print("[알림] 실패 : %s" % info)


# ─────────────────────────────────────────────────────────────
#  단독 실행 — 설정 만들기 / 시험 보내기
# ─────────────────────────────────────────────────────────────
def do_init():
    if os.path.exists(CONF_PATH):
        print("이미 있습니다 :", CONF_PATH)
        print(open(CONF_PATH, encoding="utf-8").read())
        return 0
    import random
    import string
    rnd = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    conf = json.loads(json.dumps(DEFAULT))
    conf["ntfy"]["topic"] = "baby-guard-" + rnd
    with open(CONF_PATH, "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2)
    print("만들었습니다 :", CONF_PATH)
    print()
    print(json.dumps(conf, ensure_ascii=False, indent=2))
    print()
    print("=" * 66)
    print(" 휴대폰에서 할 일 (ntfy 방식)")
    print("=" * 66)
    print("  1) 앱 설치 : Play 스토어 / App Store 에서 'ntfy'")
    print("  2) 앱에서 + 를 눌러 토픽 추가")
    print("  3) 토픽 이름에 정확히 입력 :")
    print()
    print("         %s" % conf["ntfy"]["topic"])
    print()
    print("  4) 여기서 시험 발송 : python3 notify.py --test")
    print()
    print("  ※ 토픽 이름은 비밀번호나 같다. 남이 알면 그 사람도 알림을 받거나 보낼 수 있다.")
    print("     그래서 무작위 글자를 붙였다. 발표 자료에 그대로 띄우지 말 것.")
    print("=" * 66)
    return 0


def do_test():
    conf, found = load_conf()
    if not found:
        print("notify.json 이 없습니다. 먼저 : python3 notify.py --init")
        return 1
    print("backend :", conf.get("backend"))
    n = Notifier(conf)
    if not n.enabled:
        print("설정이 덜 됐습니다. notify.json 을 확인하세요 :", CONF_PATH)
        return 1

    frame = None
    if cv2 is not None:
        import numpy as np
        frame = np.full((240, 400, 3), 40, np.uint8)
        cv2.putText(frame, "TEST", (110, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 255), 4)

    n.min_gap = 0          # 시험이니 간격 제한 없이
    n.send("시험 발송입니다. 이 메시지가 보이면 연결 성공.", frame,
           title="Baby Guard 시험")
    print("보내는 중...")
    n.close()
    print(n.summary())
    if n.n_sent:
        print("성공. 휴대폰을 확인하세요.")
        return 0
    print("실패. 아래를 확인하세요.")
    print("  - 보드가 인터넷에 나갈 수 있는가 :  ping -c 2 ntfy.sh")
    print("  - 토픽 이름 / 토큰이 맞는가 :", CONF_PATH)
    return 1


if __name__ == "__main__":
    if "--init" in sys.argv:
        sys.exit(do_init())
    if "--test" in sys.argv:
        sys.exit(do_test())
    print(__doc__)
    print("사용법 :  python3 notify.py --init    설정 파일 만들기")
    print("          python3 notify.py --test    시험 발송")
