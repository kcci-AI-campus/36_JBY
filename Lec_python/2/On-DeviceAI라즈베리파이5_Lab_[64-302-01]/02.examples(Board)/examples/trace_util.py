# -*- coding: utf-8 -*-
"""
Chrome Trace Format 트레이서 (라즈베리파이 실습용)

퓨리오사 SDK의 FURIOSA_PROFILER_OUTPUT_PATH가 만들어 주던 것과 같은 형식의
JSON을 직접 만든다. 형식은 공개 규격이라 누구나 쓸 수 있다.

보는 법:
  1) 보드에서 만들어진 trace.json을 PC로 가져온다 (MobaXterm 왼쪽 파일 탐색기에서 끌어오기)
  2) 크롬에서 https://ui.perfetto.dev 를 열고 파일을 끌어다 놓는다
     (또는 chrome://tracing 에서 Load)

쓰는 법:
    from trace_util import Tracer
    tr = Tracer('trace.json')
    while True:
        with tr.span('frame'):
            with tr.span('camera'):
                ret, frame = cap.read()
            with tr.span('inference'):
                ...
    tr.save()
"""
import json
import time
import threading
import unicodedata
from contextlib import contextmanager

try:
    import psutil
except ImportError:
    psutil = None


def _width(s):
    """터미널에서 실제로 차지하는 칸 수. 한글·한자는 2칸을 차지한다."""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)


def _ljust(s, n):
    """왼쪽 정렬. 한글 폭을 고려해 여백을 채운다."""
    return s + ' ' * max(0, n - _width(s))


def _rjust(s, n):
    """오른쪽 정렬. 한글 폭을 고려해 여백을 채운다."""
    return ' ' * max(0, n - _width(s)) + s


class Tracer:
    def __init__(self, out_path='trace.json', sample_cpu=True, sample_hz=20,
                 max_events=200000):
        self.out_path = out_path
        self.events = []
        self.max_events = max_events
        self.t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._stage_us = {}   # 단계별 누적 시간
        self._stage_n = {}    # 단계별 호출 횟수
        self._open = {}       # enter/leave 로 열어 둔 구간

        # 줄 이름 지정 (Perfetto에서 보기 좋게)
        self._meta('process_name', 1, None, {'name': 'On-Device AI pipeline'})
        self._meta('thread_name', 1, 1, {'name': 'main loop'})

        if sample_cpu:
            if psutil is None:
                print('[trace] psutil이 없어 CPU 사용률은 기록하지 않습니다.')
                print('[trace] 설치하려면: pip install psutil')
            else:
                psutil.cpu_percent(percpu=True)  # 첫 호출은 버림(기준점 잡기)
                self._thread = threading.Thread(
                    target=self._sample_loop, args=(sample_hz,), daemon=True)
                self._thread.start()

    # ---------- 내부 ----------
    def _us(self):
        return (time.perf_counter() - self.t0) * 1e6

    def _add(self, ev):
        with self._lock:
            if len(self.events) < self.max_events:
                self.events.append(ev)

    def _meta(self, name, pid, tid, args):
        ev = {'name': name, 'ph': 'M', 'pid': pid, 'args': args}
        if tid is not None:
            ev['tid'] = tid
        self._add(ev)

    def _sample_loop(self, hz):
        period = 1.0 / hz
        while not self._stop.is_set():
            per = psutil.cpu_percent(percpu=True)
            args = {('core%d' % i): v for i, v in enumerate(per)}
            self._add({'name': 'CPU per-core %', 'ph': 'C', 'ts': self._us(),
                       'pid': 1, 'args': args})
            self._add({'name': 'CPU total %', 'ph': 'C', 'ts': self._us(),
                       'pid': 1, 'args': {'total': sum(per) / len(per)}})
            self._stop.wait(period)

    # ---------- 쓰는 쪽 ----------
    @contextmanager
    def span(self, name, tid=1):
        ts = self._us()
        try:
            yield
        finally:
            dur = self._us() - ts
            self._add({'name': name, 'ph': 'X', 'ts': ts, 'dur': dur,
                       'pid': 1, 'tid': tid})
            with self._lock:
                self._stage_us[name] = self._stage_us.get(name, 0.0) + dur
                self._stage_n[name] = self._stage_n.get(name, 0) + 1

    def enter(self, name, tid=1):
        """구간 시작. with 문으로 감쌀 수 없는 자리(함수 중간 등)에서 쓴다."""
        self._open[name] = (self._us(), tid)

    def leave(self, name):
        """enter 로 연 구간을 닫는다. 열린 적 없으면 조용히 넘어간다."""
        got = self._open.pop(name, None)
        if got is None:
            return
        ts, tid = got
        dur = self._us() - ts
        self._add({'name': name, 'ph': 'X', 'ts': ts, 'dur': dur,
                   'pid': 1, 'tid': tid})
        with self._lock:
            self._stage_us[name] = self._stage_us.get(name, 0.0) + dur
            self._stage_n[name] = self._stage_n.get(name, 0) + 1

    def instant(self, name, tid=1):
        self._add({'name': name, 'ph': 'i', 'ts': self._us(),
                   'pid': 1, 'tid': tid, 's': 't'})

    def counter(self, name, args):
        self._add({'name': name, 'ph': 'C', 'ts': self._us(),
                   'pid': 1, 'args': args})

    # ---------- 마무리 ----------
    def summary(self):
        """단계별 평균 시간 표를 문자열로 만든다."""
        with self._lock:
            stages = sorted(self._stage_us.items(), key=lambda kv: -kv[1])
            n_frame = self._stage_n.get('frame', 0)
            frame_total = self._stage_us.get('frame', 0.0)
        W = 52                            # 표 전체 폭
        C1, C2, C3, C4 = 14, 11, 12, 8    # 단계 / 평균 / 합계 / 비중

        lines = []
        lines.append('')
        lines.append('=' * W)
        lines.append(' 단계별 평균 시간 (프레임 %d개)' % n_frame)
        lines.append('-' * W)
        lines.append(' ' + _ljust('단계', C1) + _rjust('평균(ms)', C2)
                     + _rjust('합계(ms)', C3) + _rjust('비중', C4))
        for name, total in stages:
            n = self._stage_n[name]
            share = (total / frame_total * 100) if frame_total and name != 'frame' else None
            lines.append(' ' + _ljust(name, C1)
                         + _rjust('%.2f' % (total / n / 1000.0), C2)
                         + _rjust('%.1f' % (total / 1000.0), C3)
                         + _rjust('-' if share is None else ('%.1f%%' % share), C4))
        if n_frame:
            fps = 1e6 / (frame_total / n_frame)
            lines.append('-' * W)
            lines.append(' 평균 FPS : %.1f   (한 프레임 %.2f ms)'
                         % (fps, frame_total / n_frame / 1000.0))
        lines.append('=' * W)
        return '\n'.join(lines)

    def save(self, print_summary=True):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            events = list(self.events)
        with open(self.out_path, 'w') as f:
            json.dump({'traceEvents': events, 'displayTimeUnit': 'ms'}, f)
        if print_summary:
            print(self.summary())
        print('[trace] 저장 완료: %s  (이벤트 %d개)' % (self.out_path, len(events)))
        print('[trace] https://ui.perfetto.dev 에 이 파일을 끌어다 놓으면 타임라인이 보입니다.')
