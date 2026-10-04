# -*- coding: utf-8 -*-
"""
웹 브라우저로 실시간 화면을 보여 주는 모듈.

팀원의 app.py 와 무엇이 다른가 (중요)
    app.py 는 추론 루프가 `generate_frames()` 안에 있었다. 즉 브라우저가
    /video_feed 를 요청해야 비로소 루프가 돈다. 그래서
        * 아무도 안 보고 있으면 카메라도 안 읽고 경보도 안 울린다
        * 두 사람이 접속하면 루프가 **두 개** 돈다 → 같은 interpreter 에
          동시에 invoke() 가 들어가는데, TFLite 인터프리터는 스레드 안전하지 않다
    감시 시스템이 "누가 보고 있을 때만" 감시하면 안 되므로 구조를 뒤집었다.

        [추론 루프 1개]  →  최신 프레임과 상태를 여기(FrameBus)에 넣는다
                                ↓ 읽기만
        [접속 A] [접속 B] [접속 C]      ← 몇 명이 봐도 추론은 한 번

주소 (팀원 app.py 와 같게 맞췄다. 대시보드가 그대로 가리킬 수 있다)
    /                 보는 페이지
    /video_feed       MJPEG 실시간 스트림   ← <img src="..."> 로 끼워 넣는 주소
    /stream.mjpg      같은 것 (이름만 다름)
    /api/status       상태 JSON
    /snapshot.jpg     지금 한 장

준비
    pip install flask

쓰는 쪽
    from web import WebServer
    web = WebServer(port=5000, token="")     # token 을 주면 ?k=토큰 이 있어야 열린다
    web.start()
    ...루프 안에서...
    web.publish(frame, status, info)
"""
import io
import sys
import time
import json
import threading

import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

try:
    from flask import Flask, Response, request, jsonify, abort
except ImportError:
    Flask = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Baby Guard System</title>
<style>
  /* 팀원 app1.py 의 디자인을 그대로 가져왔다 (팀 화면 통일) */
  :root {
    --bg-color:#0f172a; --card-bg:#1e293b; --text-main:#f8fafc; --text-sub:#94a3b8;
    --safe-color:#10b981; --detect-color:#f59e0b; --danger-color:#ef4444;
    --border-color:#334155;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background-color:var(--bg-color); color:var(--text-main);
    min-height:100vh; padding:20px; display:flex;
    justify-content:center; align-items:center;
  }
  .dashboard-container {
    width:100%; max-width:800px; background-color:var(--card-bg);
    border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,.4);
    border:1px solid var(--border-color); overflow:hidden;
    display:flex; flex-direction:column;
  }
  .header {
    padding:20px; background:#182234; border-bottom:1px solid var(--border-color);
    display:flex; justify-content:space-between; align-items:center;
  }
  .header h1 { font-size:1.25rem; font-weight:600; display:flex; align-items:center; gap:8px; }
  .live-dot {
    width:10px; height:10px; background-color:var(--danger-color);
    border-radius:50%; display:inline-block; animation:pulse 1.5s infinite;
  }
  @keyframes pulse { 0%{opacity:1} 50%{opacity:.3} 100%{opacity:1} }
  .video-wrapper {
    position:relative; background:#000; width:100%;
    display:flex; justify-content:center; align-items:center;
  }
  .video-wrapper img {
    width:100%; height:auto; max-height:480px; object-fit:contain; display:block;
  }
  .footer-panel { padding:20px; display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .stat-card {
    background:#0f172a; border:1px solid var(--border-color);
    border-radius:10px; padding:16px; text-align:center;
  }
  .stat-title { font-size:.85rem; color:var(--text-sub); margin-bottom:6px; }
  .stat-value { font-size:1.4rem; font-weight:bold; }
  .status-safe { color:var(--safe-color); }
  .status-detected { color:var(--detect-color); }
  .status-danger { color:var(--danger-color); animation:danger-blink .8s infinite alternate; }
  @keyframes danger-blink {
    from { opacity:1; transform:scale(1); }
    to   { opacity:.6; transform:scale(1.03); }
  }
  /* 우리 쪽에서 덧붙인 부분 — 같은 디자인 언어를 따른다 */
  .wide { grid-column:1 / -1; }
  .mini { display:flex; justify-content:space-around; gap:8px; flex-wrap:wrap; }
  .mini div { flex:1; min-width:80px; }
  .mini .stat-value { font-size:1.05rem; }
  .det { font-size:.95rem; color:var(--text-sub); min-height:1.2em; }
  @media (max-width:600px) { .footer-panel { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="dashboard-container">
  <div class="header">
    <h1><span class="live-dot"></span> AI 위험구역 모니터링</h1>
    <span style="font-size:.85rem;color:var(--text-sub);">Edge Device: RPi</span>
  </div>

  <div class="video-wrapper">
    <img src="/video_feed__K__" alt="Video Stream">
  </div>

  <div class="footer-panel">
    <div class="stat-card">
      <div class="stat-title">현재 보안 상태</div>
      <div id="status-text" class="stat-value status-safe">연결 중…</div>
    </div>
    <div class="stat-card">
      <div class="stat-title">누적 경보 횟수</div>
      <div id="alarm-count" class="stat-value" style="color:#38bdf8;">0 회</div>
    </div>

    <div class="stat-card wide">
      <div class="stat-title">지금 보이는 것</div>
      <div id="dets" class="det">-</div>
    </div>

    <div class="stat-card wide mini">
      <div><div class="stat-title">FPS</div><div id="fps" class="stat-value">-</div></div>
      <div><div class="stat-title">CPU 온도</div><div id="temp" class="stat-value">-</div></div>
      <div><div class="stat-title">보는 사람</div><div id="view" class="stat-value">-</div></div>
      <div><div class="stat-title">모델</div><div id="model" class="stat-value" style="font-size:.9rem;">-</div></div>
    </div>
  </div>
</div>

<script>
const K = "__K__";
const statusMap = {
  'SAFE':     { text: '정상 (SAFE)',            cls: 'status-safe' },
  'DETECTED': { text: '주의 (아기 감지)',        cls: 'status-detected' },
  'DANGER':   { text: '위험 (침입 경보 발령)',   cls: 'status-danger' }
};
setInterval(() => {
  fetch('/api/status' + K)
    .then(res => res.json())
    .then(d => {
      const info = statusMap[d.status] || { text: d.status, cls: '' };
      const st = document.getElementById('status-text');
      st.className = 'stat-value ' + info.cls;
      st.innerText = info.text;
      document.getElementById('alarm-count').innerText = d.alarm_cnt + ' 회';
      document.getElementById('fps').innerText   = d.fps ? d.fps.toFixed(1) : '-';
      document.getElementById('temp').innerText  = d.temp ? d.temp.toFixed(1) + ' ℃' : '-';
      document.getElementById('view').innerText  = (d.viewers || 0) + ' 명';
      document.getElementById('model').innerText = d.model || '-';
      document.getElementById('dets').innerText  =
        (d.detections || []).map(x => x.name + ' ' + Math.round(x.conf * 100) + '%')
                            .join('   ·   ') || '없음';
    })
    .catch(err => console.error('상태 수신 오류:', err));
}, 1000);
</script>
</body>
</html>"""


# 보드 여러 대를 한 화면에 보는 쪽. 위 한 대짜리와 디자인은 같다.
TEAM_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Team 10 — AI Baby Guard</title>
<style>
  :root {
    --bg-color:#0f172a; --card-bg:#1e293b; --text-main:#f8fafc; --text-sub:#94a3b8;
    --safe-color:#10b981; --detect-color:#f59e0b; --danger-color:#ef4444;
    --border-color:#334155;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background-color:var(--bg-color); color:var(--text-main);
    min-height:100vh; padding:20px;
  }
  .page { max-width:1500px; margin:0 auto; }
  .header {
    padding:16px 20px; background:#182234; border:1px solid var(--border-color);
    border-radius:14px; margin-bottom:18px;
    display:flex; justify-content:space-between; align-items:center;
  }
  .header h1 { font-size:1.25rem; font-weight:600; display:flex; align-items:center; gap:8px; }
  .live-dot { width:10px; height:10px; background-color:var(--danger-color);
              border-radius:50%; display:inline-block; animation:pulse 1.5s infinite; }
  @keyframes pulse { 0%{opacity:1} 50%{opacity:.3} 100%{opacity:1} }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:18px; }
  .board {
    background-color:var(--card-bg); border:1px solid var(--border-color);
    border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,.4);
    overflow:hidden; display:flex; flex-direction:column;
  }
  .board.alert { border-color:var(--danger-color); box-shadow:0 0 0 3px var(--danger-color); }
  .board-title {
    padding:12px 16px; background:#182234; border-bottom:1px solid var(--border-color);
    display:flex; justify-content:space-between; align-items:center; font-weight:600;
  }
  .badge { font-size:.78rem; font-weight:600; padding:3px 10px; border-radius:20px; background:#334155; }
  .badge.on  { background:var(--safe-color); color:#04261a; }
  .badge.off { background:#64748b; }
  .video-wrapper { position:relative; background:#000; width:100%; min-height:190px;
                   display:flex; justify-content:center; align-items:center; }
  .video-wrapper img { width:100%; height:auto; max-height:400px; object-fit:contain; display:block; }
  .offline { color:var(--text-sub); font-size:.95rem; padding:56px 20px; text-align:center; }
  .footer-panel { padding:14px; display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .stat-card { background:#0f172a; border:1px solid var(--border-color);
               border-radius:10px; padding:12px; text-align:center; }
  .stat-title { font-size:.78rem; color:var(--text-sub); margin-bottom:5px; }
  .stat-value { font-size:1.25rem; font-weight:bold; }
  .status-safe { color:var(--safe-color); }
  .status-detected { color:var(--detect-color); }
  .status-danger { color:var(--danger-color); animation:danger-blink .8s infinite alternate; }
  @keyframes danger-blink { from{opacity:1;transform:scale(1)} to{opacity:.6;transform:scale(1.03)} }
  .wide { grid-column:1 / -1; }
  .det { font-size:.88rem; color:var(--text-sub); min-height:1.2em; }
  @media (max-width:600px) { .footer-panel { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1><span class="live-dot"></span> AI 위험구역 모니터링 — Team 10</h1>
    <span style="font-size:.85rem;color:var(--text-sub);">Edge Device: Raspberry Pi 5</span>
  </div>
  <div class="grid" id="grid"></div>
</div>
<script>
const BOARDS = __BOARDS__;
const statusMap = {
  'SAFE':     { text:'정상 (SAFE)',          cls:'status-safe' },
  'DETECTED': { text:'주의 (아기 감지)',      cls:'status-detected' },
  'DANGER':   { text:'위험 (침입 경보 발령)', cls:'status-danger' }
};

// 카드를 자바스크립트로 만든다. 보드 수가 바뀌어도 HTML 을 고칠 필요가 없다.
document.getElementById('grid').innerHTML = BOARDS.map((b, i) => `
  <div class="board" id="board-${i}">
    <div class="board-title">
      <span>${b.name}</span><span class="badge" id="live-${i}">연결 중</span>
    </div>
    <div class="video-wrapper" id="vw-${i}">
      <img src="${b.base}/video_feed${b.k}" alt="${b.name}"
           onerror="document.getElementById('vw-${i}').innerHTML=
             '<div class=\\'offline\\'>연결 없음<br><small>이 보드에서 프로그램이 돌고 있는지 확인하세요</small></div>';">
    </div>
    <div class="footer-panel">
      <div class="stat-card"><div class="stat-title">현재 보안 상태</div>
        <div id="status-${i}" class="stat-value">-</div></div>
      <div class="stat-card"><div class="stat-title">누적 경보 횟수</div>
        <div id="alarm-${i}" class="stat-value" style="color:#38bdf8;">-</div></div>
      <div class="stat-card wide"><div class="stat-title">지금 보이는 것</div>
        <div id="dets-${i}" class="det">-</div></div>
    </div>
  </div>`).join('');

function refresh(i) {
  fetch(BOARDS[i].base + '/api/status' + BOARDS[i].k, { cache:'no-store' })
    .then(r => r.json())
    .then(d => {
      const live = document.getElementById('live-' + i);
      live.className = 'badge on'; live.innerText = '접속됨';
      const info = statusMap[d.status] || { text:d.status, cls:'' };
      const st = document.getElementById('status-' + i);
      st.className = 'stat-value ' + info.cls; st.innerText = info.text;
      document.getElementById('alarm-' + i).innerText = (d.alarm_cnt || 0) + ' 회';
      const de = document.getElementById('dets-' + i);
      de.innerText = d.detections
        ? (d.detections.map(x => x.name + ' ' + Math.round(x.conf*100) + '%').join('   ·   ') || '없음')
        : '-';
      document.getElementById('board-' + i).classList.toggle('alert', d.status === 'DANGER');
    })
    .catch(() => {
      const live = document.getElementById('live-' + i);
      live.className = 'badge off'; live.innerText = '연결 없음';
    });
}
setInterval(() => BOARDS.forEach((_, i) => refresh(i)), 1000);
BOARDS.forEach((_, i) => refresh(i));
</script>
</body>
</html>"""


class WebServer:
    """추론 루프가 넣어 준 최신 프레임을 여러 접속자에게 그대로 나눠 준다."""

    def __init__(self, port=5000, token="", quality=70, model_name="",
                 verbose=True, boards=None):
        # boards 를 주면 '/' 가 여러 대를 보여 주는 팀 화면이 된다.
        #   [(이름, "http://10.10.15.61:5000"), ...]
        self.boards = boards or []
        self.port = int(port)
        self.token = token or ""
        self.quality = int(quality)
        self.model_name = model_name
        self.verbose = verbose
        self.enabled = Flask is not None
        if not self.enabled and verbose:
            print("[웹] flask 가 없어 웹 화면을 끕니다.  pip install flask")
            return

        self._jpg = None                       # 최신 프레임(JPEG 바이트)
        self._seq = 0                          # 프레임 번호. 이게 바뀌면 새 그림
        self._cv = threading.Condition()       # 새 프레임을 기다리는 장치
        self._status = {"status": "SAFE", "alarm_cnt": 0, "fps": 0.0,
                        "detections": [], "temp": None, "model": model_name,
                        "danger_info": None}
        self.n_view = 0                        # 지금 보고 있는 사람 수
        self._app = self._build()
        self._th = None

    # ---------- 추론 루프가 부른다 ----------
    def publish(self, frame, status="SAFE", alarm_cnt=0, fps=0.0,
                detections=None, temp=None, danger_info=None):
        """최신 화면과 상태를 올린다. 접속자가 없으면 인코딩도 건너뛴다."""
        if not self.enabled:
            return
        with self._cv:
            self._status.update({"status": status, "alarm_cnt": alarm_cnt,
                                 "fps": float(fps), "temp": temp,
                                 "danger_info": danger_info,
                                 "detections": detections or []})
            # 아무도 안 보면 JPEG 로 만들 필요가 없다 (저전력)
            if self.n_view > 0:
                ok, buf = cv2.imencode(".jpg", frame,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if ok:
                    self._jpg = buf.tobytes()
                    self._seq += 1
                    self._cv.notify_all()

    @staticmethod
    def my_ip():
        """이 보드의 랜 IP. 밖으로 나가는 소켓을 열어 보면 알 수 있다.

        실제로 보내지는 않는다. 어느 랜카드로 나갈지만 커널에 물어보는 것이다.
        hostname -I 는 여러 개를 주고 순서도 제멋대로라 이 방법이 확실하다.
        """
        import socket
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.connect(("8.8.8.8", 80))
            ip = sk.getsockname()[0]
            sk.close()
            return ip
        except OSError:
            return None

    def start(self):
        if not self.enabled:
            return
        self._th = threading.Thread(target=self._serve, daemon=True)
        self._th.start()
        time.sleep(0.3)
        if self.verbose:
            k = ("?k=" + self.token) if self.token else ""
            ip = self.my_ip() or "<보드IP>"
            print()
            print("=" * 58)
            print("  브라우저 주소창에 이것을 붙여 넣으세요")
            print()
            print("      http://%s:%d/%s" % (ip, self.port, k))
            print()
            print("=" * 58)
            print("     같은 네트워크의 PC·휴대폰이면 된다.")
            if self.boards:
                print("     [팀 화면] 보드 %d대를 한 화면에 표시합니다." % len(self.boards))
            print("     밖에서(LTE) 보려면 다른 터미널에서 :")
            print("       cloudflared tunnel --url http://localhost:%d" % self.port)
            print("       -> 나온 https 주소를 --web-url 로 다시 줄 것")
            if not self.token:
                print("     ※ 비밀번호가 없다. 주소를 아는 사람은 누구나 본다.")
                print("        걸려면 --web-token 내가정한암호")

    def summary(self):
        if not self.enabled:
            return "  웹            : 사용 안 함"
        return "  웹            : 포트 %d (마지막 접속자 %d명)" % (self.port, self.n_view)

    # ---------- 내부 ----------
    def _check(self):
        if self.token and request.args.get("k", "") != self.token:
            abort(403)

    def _build(self):
        app = Flask(__name__)
        # 기본 로그가 프레임마다 찍혀 터미널을 덮으므로 끈다
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        @app.route("/")
        def index():
            self._check()
            k = ("?k=" + self.token) if self.token else ""
            if self.boards:
                import json as _json
                bs = [{"name": n, "base": b.rstrip("/"), "k": k}
                      for n, b in self.boards]
                return TEAM_PAGE.replace("__BOARDS__",
                                         _json.dumps(bs, ensure_ascii=False))
            return HTML_PAGE.replace("__K__", k)

        def mjpeg():
            """새 프레임이 올라올 때까지 기다렸다가 보낸다. 바쁜 대기를 하지 않는다."""
            last = -1
            with self._cv:
                self.n_view += 1
            try:
                while True:
                    with self._cv:
                        if self._seq == last:
                            self._cv.wait(timeout=2.0)
                        if self._jpg is None or self._seq == last:
                            continue
                        last, data = self._seq, self._jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + data + b"\r\n")
            finally:
                with self._cv:
                    self.n_view = max(0, self.n_view - 1)

        @app.route("/video_feed")
        @app.route("/stream.mjpg")
        def video_feed():
            self._check()
            return Response(mjpeg(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/api/status")
        def api_status():
            self._check()
            with self._cv:
                st = dict(self._status)
            st["viewers"] = self.n_view
            resp = jsonify(st)
            # 다른 보드의 대시보드 페이지가 이 상태를 읽을 수 있게 허용한다.
            #   영상(<img>)은 그냥 되지만 fetch() 는 브라우저가 출처를 따진다(CORS).
            #   읽기 전용 정보라 전체 허용해도 문제없다.
            resp.headers["Access-Control-Allow-Origin"] = "*"
            return resp

        @app.route("/snapshot.jpg")
        def snapshot():
            self._check()
            with self._cv:
                # 보는 사람이 없으면 최신 JPEG 이 없을 수 있다. 한 장만 만들게 표시
                data = self._jpg
            if data is None:
                return Response("아직 화면이 없습니다. /video_feed 를 한 번 열어 주세요.",
                                status=503, mimetype="text/plain; charset=utf-8")
            return Response(data, mimetype="image/jpeg")

        return app

    def _serve(self):
        try:
            self._app.run(host="0.0.0.0", port=self.port,
                          threaded=True, debug=False, use_reloader=False)
        except Exception as e:
            print("[웹] 서버를 띄울 수 없습니다 : %s" % e)
            self.enabled = False


if __name__ == "__main__":
    print(__doc__)
    print("이 파일은 단독으로 쓰지 않는다. detect_baby.py --web 으로 실행한다.")
