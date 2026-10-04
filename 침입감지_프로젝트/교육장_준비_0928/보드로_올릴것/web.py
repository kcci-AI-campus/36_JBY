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
    from flask import Flask, Response, request, jsonify, abort, make_response, redirect
except ImportError:
    Flask = None


HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Baby Guard · 위험구역 모니터링</title>
    <style>
        /* ── 디자인 토큰 ────────────────────────────────── */
        :root {
            --bg: #070b14;
            --bg-glow-1: rgba(56, 189, 248, 0.16);
            --bg-glow-2: rgba(239, 68, 68, 0.12);
            --panel: rgba(23, 32, 51, 0.72);
            --panel-solid: #131c2e;
            --border: rgba(148, 163, 184, 0.16);
            --text: #e8edf6;
            --text-sub: #8b9bb4;
            --safe: #22d39a;
            --detect: #f5a623;
            --danger: #ff4d4d;
            --info: #38bdf8;
            --radius: 20px;
            --shadow: 0 24px 60px -20px rgba(0, 0, 0, 0.75);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; }
        body {
            font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic", Roboto, sans-serif;
            /* 배경 글로우를 단일 static 배경으로 흡수 (전체화면 fixed 레이어 제거) */
            background-color: var(--bg);
            background-image:
                radial-gradient(60% 55% at 15% 10%, var(--bg-glow-1), transparent 60%),
                radial-gradient(55% 50% at 90% 95%, var(--bg-glow-2), transparent 60%);
            background-attachment: fixed;
            color: var(--text);
            min-height: 100vh;
            padding: 32px 20px;
            display: flex;
            justify-content: center;
            align-items: flex-start;   /* center 이면 화면보다 긴 페이지의 윗부분이 스크롤로도 안 보이게 잘린다 (9/28 수정) */
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden;
        }

        /* ── 레이아웃 ───────────────────────────────────── */
        /* 성능: backdrop-filter(실시간 블러) 제거 → 불투명 패널 */
        .dashboard {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 900px;
            background: var(--panel-solid);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* ── 상단 헤더 ──────────────────────────────────── */
        .header {
            padding: 20px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            border-bottom: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(56,189,248,0.06), transparent);
        }
        .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
        .brand-icon {
            width: 42px; height: 42px; flex: 0 0 auto;
            display: grid; place-items: center;
            border-radius: 13px;
            background: linear-gradient(145deg, #1e40af, #0ea5e9);
            box-shadow: 0 8px 20px -6px rgba(14, 165, 233, 0.7);
            font-size: 1.25rem;
        }
        .brand-text h1 { font-size: 1.12rem; font-weight: 700; letter-spacing: -0.01em; }
        .brand-text p { font-size: 0.76rem; color: var(--text-sub); margin-top: 2px; }

        .live-pill {
            display: inline-flex; align-items: center; gap: 7px;
            padding: 7px 14px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #ffd7d7;
            background: rgba(239, 68, 68, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.35);
            flex: 0 0 auto;
        }
        .live-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--danger);
            /* 성능: box-shadow 애니메이션(리페인트) → opacity만 합성 */
            animation: pulse 1.6s infinite;
            will-change: opacity;
        }
        @keyframes pulse {
            0%   { opacity: 1; }
            70%  { opacity: 0.35; }
            100% { opacity: 1; }
        }

        /* ── 경보 배너 (평소엔 숨김) ───────────────────── */
        .alert-banner {
            max-height: 0;
            overflow: hidden;
            opacity: 0;
            transition: max-height .35s ease, opacity .35s ease;
            background: linear-gradient(90deg, #b91c1c, #ef4444, #b91c1c);
            color: #fff;
            text-align: center;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.06em;
        }
        .alert-banner span { display: block; padding: 12px; }
        body.is-danger .alert-banner { max-height: 60px; opacity: 1; }
        /* 성능: 텍스트 깜빡임(재레이아웃 유발) 제거 → 정적 배너만 표시 */

        /* ── 영상 영역 ──────────────────────────────────── */
        .video-wrapper {
            position: relative;
            background: #04060c;
            display: flex;
            justify-content: center;
            align-items: flex-start;   /* center 이면 화면보다 긴 페이지의 윗부분이 스크롤로도 안 보이게 잘린다 (9/28 수정) */
            line-height: 0;
            /* 성능: 영상 갱신이 바깥 레이아웃에 영향을 주지 않도록 격리 */
            contain: content;
        }
        .video-wrapper img {
            width: 100%;
            height: auto;
            max-height: 520px;
            object-fit: contain;
            display: block;
        }
        /* 영상 오버레이 테두리 (성능: inset box-shadow 90px 대신 가벼운 border) */
        .video-wrapper::after {
            content: "";
            position: absolute; inset: 0;
            pointer-events: none;
            border: 2px solid transparent;
            border-radius: 2px;
        }
        body.is-danger .video-wrapper::after { border-color: rgba(255, 77, 77, 0.9); }

        /* 영상 좌상단 상태 칩 */
        .video-chip {
            position: absolute;
            top: 14px; left: 14px;
            display: inline-flex; align-items: center; gap: 8px;
            padding: 7px 14px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            /* 성능: backdrop-filter 제거 → 충분히 불투명한 배경 */
            background: rgba(6, 10, 18, 0.82);
            border: 1px solid var(--border);
            color: var(--text);
        }
        .video-chip .dot {
            width: 9px; height: 9px; border-radius: 50%; background: var(--safe);
        }

        /* 영상 좌상단 FPS / 경보 요약 오버레이 */
        .video-hud {
            position: absolute;
            top: 14px; right: 14px;
            display: inline-flex; align-items: center; gap: 10px;
            padding: 7px 14px;
            border-radius: 10px;
            font-size: 0.78rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            background: rgba(6, 10, 18, 0.82);
            border: 1px solid var(--border);
            color: var(--text);
        }
        .video-hud .hud-sep { color: var(--text-sub); font-weight: 400; }
        .video-hud #hud-alarm { color: var(--danger); }

        /* ── 통계 패널 ──────────────────────────────────── */
        .stats {
            padding: 22px 24px 26px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
        }
        .stat {
            position: relative;
            padding: 18px 16px;
            border-radius: 14px;
            background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01));
            border: 1px solid var(--border);
            overflow: hidden;
            transition: transform .25s ease, border-color .25s ease;
        }
        .stat:hover { transform: translateY(-2px); border-color: rgba(148,163,184,0.32); }
        /* 영상 갱신 중 리페인트 최소화 */
        .stat, .header, .footer { content-visibility: auto; }
        .stat .label {
            display: flex; align-items: center; gap: 6px;
            font-size: 0.76rem; color: var(--text-sub);
            margin-bottom: 10px; font-weight: 600;
        }
        .stat .value {
            font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;
            font-variant-numeric: tabular-nums;
        }
        .stat .value .unit { font-size: .9rem; font-weight: 600; color: var(--text-sub); }
        .stat .sub { font-size: 0.7rem; color: var(--text-sub); margin-top: 4px; }
        /* 좌측 컬러 액센트 바 */
        .stat::before {
            content: "";
            position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
            background: var(--accent, var(--info));
        }
        .stat.state    { --accent: var(--safe); }
        .stat.alarm    { --accent: var(--danger); }
        .stat.fps      { --accent: var(--info); }

        /* 상태별 색상 */
        /* 성능: 상태 텍스트 깜빡임 애니메이션 제거 (색상으로만 구분) */
        .v-safe     { color: var(--safe); }
        .v-detected { color: var(--detect); }
        .v-danger   { color: var(--danger); }

        /* 모션 최소화 선호 시 모든 애니메이션 중단 */
        @media (prefers-reduced-motion: reduce) {
            * { animation: none !important; transition: none !important; }
        }

        /* ── 하단 푸터 ──────────────────────────────────── */
        .footer {
            display: flex; justify-content: space-between; align-items: center;
            padding: 14px 24px;
            border-top: 1px solid var(--border);
            font-size: 0.74rem; color: var(--text-sub);
            background: rgba(0,0,0,0.18);
        }
        .footer .clock { font-variant-numeric: tabular-nums; color: var(--text); font-weight: 600; }

        /* ── 3색 LED 상태 패널 ───────────────────────────── */
        .led-panel {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 14px;
        }
        .led-item {
            display: flex; align-items: center; gap: 10px;
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.02);
            transition: background .25s ease, border-color .25s ease, box-shadow .25s ease;
        }
        .led-bulb {
            width: 16px; height: 16px; border-radius: 50%;
            flex: 0 0 auto;
            background: #2a3550;
            border: 1px solid rgba(255,255,255,0.12);
            transition: background .2s ease, box-shadow .2s ease;
        }
        .led-item .led-name { font-size: 0.82rem; font-weight: 700; }
        .led-item .led-desc { font-size: 0.68rem; color: var(--text-sub); margin-top: 1px; }
        .led-item[data-color="green"].on {
            border-color: rgba(34, 211, 154, 0.55);
            background: rgba(34, 211, 154, 0.10);
        }
        .led-item[data-color="green"].on .led-bulb {
            background: var(--safe); box-shadow: 0 0 12px 2px rgba(34, 211, 154, 0.75);
        }
        .led-item[data-color="blue"].on {
            border-color: rgba(245, 166, 35, 0.55);
            background: rgba(245, 166, 35, 0.10);
        }
        .led-item[data-color="blue"].on .led-bulb {
            background: var(--detect); box-shadow: 0 0 12px 2px rgba(245, 166, 35, 0.75);
        }
        .led-item[data-color="red"].on {
            border-color: rgba(255, 77, 77, 0.55);
            background: rgba(255, 77, 77, 0.10);
        }
        .led-item[data-color="red"].on .led-bulb {
            background: var(--danger); box-shadow: 0 0 12px 2px rgba(255, 77, 77, 0.8);
        }

        /* ── 클래스별 검출 현황 ──────────────────────────── */
        .class-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 14px;
        }
        .class-card {
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01));
            border-top: 3px solid var(--c, var(--info));
        }
        .class-card .cc-name { font-size: 0.72rem; color: var(--text-sub); font-weight: 600; }
        .class-card .cc-ko { font-size: 1rem; font-weight: 800; margin-top: 2px; color: var(--c, var(--text)); }
        .class-card .cc-stat { font-size: 0.68rem; color: var(--text-sub); margin-top: 6px; }
        .class-card .cc-stat b { color: var(--text); font-variant-numeric: tabular-nums; }
        .class-card.hit { box-shadow: 0 0 0 1px var(--c) inset; }

        /* ── 이벤트 로그 ─────────────────────────────────── */
        .log-panel { max-height: 320px; overflow-y: auto; padding-right: 4px; }
        .log-panel::-webkit-scrollbar { width: 6px; }
        .log-panel::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.3); border-radius: 3px; }
        .log-empty { font-size: 0.78rem; color: var(--text-sub); padding: 22px 4px; text-align: center; }
        .log-item {
            display: flex; gap: 10px;
            padding: 11px 12px;
            border-radius: 10px;
            border: 1px solid var(--border);
            border-left: 4px solid var(--lv, var(--info));
            background: rgba(255,255,255,0.02);
            margin-bottom: 8px;
        }
        .log-item.danger  { --lv: var(--danger); }
        .log-item.caution { --lv: var(--detect); }
        .log-item.safe    { --lv: var(--safe); }
        .log-item.info    { --lv: var(--info); }
        .log-body { min-width: 0; }
        .log-head { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
        .log-title { font-size: 0.8rem; font-weight: 700; }
        .log-item.danger  .log-title { color: #ffb4b4; }
        .log-item.caution .log-title { color: #ffd79a; }
        .log-item.safe    .log-title { color: #8ff0cd; }
        .log-time { font-size: 0.68rem; color: var(--text-sub); font-variant-numeric: tabular-nums; flex: 0 0 auto; }
        .log-detail { font-size: 0.72rem; color: var(--text-sub); margin-top: 3px; line-height: 1.45; }

        /* ── 반응형 ─────────────────────────────────────── */
        @media (max-width: 640px) {
            body { padding: 16px 12px; }
            .stats { grid-template-columns: 1fr; }
            .led-panel { grid-template-columns: 1fr; }
            .class-grid { grid-template-columns: repeat(2, 1fr); }
            .brand-text h1 { font-size: 1rem; }
            .stat .value { font-size: 1.35rem; }
            .live-pill .txt { display: none; }
        }
    
        /* 우리 배선은 가운데가 파랑이다 (LED 3개가 따로 떨어져 있어
           빨강+초록으로 노랑을 만들 수 없다) */
        .led-item[data-color="blue"] .led-bulb { background:#38bdf8; }
        .led-item[data-color="blue"].on .led-bulb {
            background:#38bdf8; box-shadow:0 0 14px #38bdf8; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <div class="brand">
                <div class="brand-icon">🛡️</div>
                <div class="brand-text">
                    <h1>AI 위험구역 모니터링</h1>
                    <p>실시간 침입 감지 · Edge Device (RPi)</p>
                </div>
            </div>
            <div class="live-pill"><span class="live-dot"></span><span class="txt">LIVE</span></div>
        </div>

        <div class="alert-banner"><span>⚠ 위험 감지 — 아기가 위험 물체에 접근했습니다</span></div>

        <div class="video-wrapper">
            <div class="video-chip"><span class="dot" id="chip-dot"></span><span id="chip-text">정상</span></div>
            <div class="video-hud"><span>FPS <span id="hud-fps">0.0</span></span><span class="hud-sep">|</span><span>alarm <span id="hud-alarm">0</span></span></div>
            <img src="/video_feed__K__" alt="실시간 영상 스트림">
        </div>

        <div class="stats">
            <div class="stat state">
                <div class="label">🔒 현재 보안 상태</div>
                <div id="status-text" class="value v-safe">정상 (SAFE)</div>
                <div class="sub">아기 · 위험물체 실시간 판정</div>
            </div>
            <div class="stat alarm">
                <div class="label">🚨 누적 경보</div>
                <div class="value"><span id="alarm-count">0</span> <span class="unit">회</span></div>
                <div class="sub">위험 반경 침입 감지 횟수</div>
            </div>
            <div class="stat fps">
                <div class="label">⚡ 처리 성능</div>
                <div class="value"><span id="fps-text">0.0</span> <span class="unit">FPS</span></div>
                <div class="sub">초당 추론 프레임</div>
            </div>
        </div>

        <div class="stats" style="grid-template-columns: 1fr;">
            <div class="stat">
                <div class="label">💡 3색 LED 상태</div>
                <div class="led-panel">
                    <div class="led-item" data-color="green" id="led-green">
                        <span class="led-bulb"></span>
                        <span><span class="led-name">GREEN</span><div class="led-desc">정상 · 안전</div></span>
                    </div>
                    <div class="led-item" data-color="blue" id="led-yellow">
                        <span class="led-bulb"></span>
                        <span><span class="led-name">BLUE</span><div class="led-desc">아기 감지 · 주의</div></span>
                    </div>
                    <div class="led-item" data-color="red" id="led-red">
                        <span class="led-bulb"></span>
                        <span><span class="led-name">RED</span><div class="led-desc">위험 · 경보</div></span>
                    </div>
                </div>
            </div>

            <div class="stat">
                <div class="label">🧠 클래스별 감지 현황</div>
                <div class="class-grid" id="class-grid"></div>
            </div>

            <div class="stat alarm">
                <div class="label">📋 위험구역 침입 기록 <span id="log-count" class="unit">0 건</span></div>
                <div class="log-panel" id="log-panel">
                    <div class="log-empty">· 기록 없음 ·</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <span>Baby Guard · On-Device AI <span id="ft-model">-</span> · <span id="ft-temp">-</span> · 보는 사람 <span id="ft-view">-</span></span>
            <span class="clock" id="clock">--:--:--</span>
        </div>
    </div>

    <script>
        const K = "__K__";
        const statusMap = {
            'SAFE':     { text: '정상 (SAFE)',       cls: 'v-safe',     chip: '정상',       danger: false },
            'DETECTED': { text: '주의 (아기 감지)',   cls: 'v-detected', chip: '아기 감지',   danger: false },
            'DANGER':   { text: '위험 (경보 발령)',   cls: 'v-danger',   chip: '위험 경보',   danger: true  }
        };
        const chipColors = { 'v-safe': '#22d39a', 'v-detected': '#f5a623', 'v-danger': '#ff4d4d' };
        const ledColors  = { 'v-safe': 'green', 'v-detected': 'yellow', 'v-danger': 'red' };
        const classMeta  = {
            'baby':   { ko: 'Baby (아기)',      c: '#00c8ff' },
            'adult':  { ko: 'Adult (성인)',     c: '#c8c8c8' },
            'knife':  { ko: 'Knife (칼)',       c: '#ff4d4d' },
            'outlet': { ko: 'Outlet (콘센트)', c: '#ff4d4d' }
        };
        const classOrder = ['baby', 'adult', 'knife', 'outlet'];

        // 성능: DOM 요소 캐시 (매 폴링마다 getElementById 호출 제거)
        const statusEl  = document.getElementById('status-text');
        const chipText  = document.getElementById('chip-text');
        const chipDot   = document.getElementById('chip-dot');
        const alarmEl   = document.getElementById('alarm-count');
        const fpsEl     = document.getElementById('fps-text');
        const clockEl   = document.getElementById('clock');
        const hudFps    = document.getElementById('hud-fps');
        const hudAlarm  = document.getElementById('hud-alarm');
        const ledEls    = { green: document.getElementById('led-green'),
                            yellow: document.getElementById('led-yellow'),
                            red: document.getElementById('led-red') };
        const logPanel  = document.getElementById('log-panel');
        const logCount  = document.getElementById('log-count');
        const state = { lastStatus: null, lastAlarm: null, lastFps: null, lastLed: null,
                        lastCounts: null, lastLogLen: -1, countsInit: false };

        // 클래스 카드 초기 구성
        const classCards = {};
        classOrder.forEach(name => {
            const meta = classMeta[name] || { ko: name, c: '#38bdf8' };
            const el = document.createElement('div');
            el.className = 'class-card';
            el.style.setProperty('--c', meta.c);
            el.innerHTML = '<div class="cc-name">CLASS ' + classOrder.indexOf(name) + '</div>'
                         + '<div class="cc-ko">' + meta.ko + '</div>'
                         + '<div class="cc-stat">검출 <b>0</b> 회</div>'
                         + '<div class="cc-stat cc-conf">지금 신뢰도 <b>–</b></div>';
            document.getElementById('class-grid').appendChild(el);
            classCards[name] = el;
        });

        function tickClock() {
            const d = new Date();
            const p = n => String(n).padStart(2, '0');
            if (state.ft) {
                    document.getElementById('ft-model').textContent = state.ft.model || '-';
                    document.getElementById('ft-temp').textContent =
                        state.ft.temp ? state.ft.temp.toFixed(1) + ' ℃' : '-';
                    document.getElementById('ft-view').textContent = state.ft.viewers;
                }
            clockEl.textContent =
                p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
        }
        tickClock();
        setInterval(tickClock, 1000);

        async function poll() {
            try {
                const res = await fetch('/api/status' + K);
                const data = await res.json();
                const info = statusMap[data.status] || { text: data.status, cls: '', chip: data.status, danger: false };

                // 성능: 값이 실제로 바뀔 때만 DOM 갱신 (불필요한 리플로우 방지)
                if (state.lastStatus !== data.status) {
                    state.lastStatus = data.status;
                    statusEl.className = 'value ' + info.cls;
                    statusEl.textContent = info.text;
                    chipText.textContent = info.chip;
                    const c = chipColors[info.cls] || '#22d39a';
                    chipDot.style.background = c;
                    document.body.classList.toggle('is-danger', !!info.danger);
                }
                const led = ledColors[info.cls] || 'green';
                if (state.lastLed !== led) {
                    state.lastLed = led;
                    for (const k in ledEls) ledEls[k].classList.toggle('on', k === led);
                }
                if (state.lastAlarm !== data.alarm_cnt) {
                    state.lastAlarm = data.alarm_cnt;
                    alarmEl.textContent = data.alarm_cnt;
                    hudAlarm.textContent = data.alarm_cnt;
                }
                state.ft = { model: data.model, temp: data.temp, viewers: data.viewers };
                const fv = (data.fps || 0).toFixed(1);
                if (state.lastFps !== fv) {
                    state.lastFps = fv;
                    fpsEl.textContent = fv;
                    hudFps.textContent = fv;
                }
                // 클래스별 지금 신뢰도 (이번 프레임에서 그 클래스 박스 중 가장 높은 값, 없으면 –)  9/28 추가
                const best = {};
                (data.detections || []).forEach(d => { if (!(d.name in best) || d.conf > best[d.name]) best[d.name] = d.conf; });
                classOrder.forEach(name => {
                    const b = classCards[name].querySelector('.cc-conf b');
                    const t = (name in best) ? Math.round(best[name] * 100) + ' %' : '–';
                    if (b.textContent !== t) b.textContent = t;
                });
                // 클래스별 감지 현황
                const counts = data.class_counts || {};
                const ckey = JSON.stringify(counts);
                if (state.lastCounts !== ckey) {
                    state.lastCounts = ckey;
                    classOrder.forEach(name => {
                        const el = classCards[name];
                        const n = counts[name] || 0;
                        el.querySelector('.cc-stat b').textContent = n;
                        el.classList.toggle('hit', n > (state.countsInit ? (el._prev || 0) : -1));
                        el._prev = n;
                    });
                    state.countsInit = true;
                }
                // 이벤트 로그 (길이 기반으로 변경 시에만 재렌더)
                const logs = data.logs || [];
                if (state.lastLogLen !== logs.length || state.lastLogTop !== (logs[0] && logs[0].time + logs[0].title)) {
                    state.lastLogLen = logs.length;
                    state.lastLogTop = logs[0] && logs[0].time + logs[0].title;
                    logCount.textContent = logs.length + ' 건';
                    if (!logs.length) {
                        logPanel.innerHTML = '<div class="log-empty">· 기록 없음 ·</div>';
                    } else {
                        logPanel.innerHTML = logs.map(e =>
                            '<div class="log-item ' + e.level + '">'
                          + '<div class="log-body">'
                          + '<div class="log-head"><span class="log-title">' + e.title + '</span>'
                          + '<span class="log-time">' + e.time + '</span></div>'
                          + '<div class="log-detail">' + e.detail + '</div>'
                          + '</div></div>').join('');
                    }
                }
            } catch (e) {
                console.error('상태 수신 오류:', e);
            }
        }
        poll();
        setInterval(poll, 1000);
    </script>
</body>
</html>
"""


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
                        "danger_info": None, "class_counts": {}, "led": "green"}
        self._logs = []                        # 위험 기록. 최근 30건만 들고 있는다
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
            dets = detections or []

            # 클래스별로 몇 개를 보고 있는지 (화면의 '클래스별 감지 현황' 칸)
            counts = {}
            for d in dets:
                counts[d["name"]] = counts.get(d["name"], 0) + 1

            # LED 색. 보드의 실제 LED 와 같은 규칙을 쓴다.
            #   (가운데 자리는 우리 배선에서 파랑이지만, 화면 코드가 쓰는
            #    이름은 yellow 라서 값은 그대로 두고 화면에서만 BLUE 로 보여 준다)
            led = "red" if status == "DANGER" else (
                  "yellow" if status == "DETECTED" else "green")

            # 위험 기록. 상태가 DANGER 로 '바뀌는 순간' 에만 한 줄 남긴다.
            #   매 프레임 남기면 초당 수십 줄이 쌓여 아무 쓸모가 없다.
            if status == "DANGER" and self._status.get("status") != "DANGER":
                self._logs.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "level": "danger",
                    "title": "위험구역 침입",
                    "detail": danger_info or "아기가 위험물에 접근",
                })
                del self._logs[30:]          # 최근 30건만 들고 있는다

            self._status.update({"status": status, "alarm_cnt": alarm_cnt,
                                 "fps": float(fps), "temp": temp,
                                 "danger_info": danger_info,
                                 "detections": dets,
                                 "class_counts": counts,
                                 "led": led})
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
    # 쿠키 이름. 토큰을 주소창에서 치우기 위해 쓴다.
    COOKIE = "bg_k"

    def _check(self):
        """암호를 확인한다. 주소의 ?k= 또는 쿠키 둘 중 하나면 통과.

        왜 쿠키를 쓰나 :
          ?k=암호 가 주소창에 그대로 남으면
            - 화면 캡처에 찍힌다 (발표 자료에 그대로 들어간다)
            - 브라우저 방문기록·자동완성에 남는다
            - 링크를 보여 줄 때 같이 새어 나간다
          그래서 첫 접속 때만 ?k= 로 확인하고, 쿠키로 옮긴 뒤 주소창에서 지운다.
        """
        if not self.token:
            return
        if request.args.get("k", "") == self.token:
            return
        if request.cookies.get(self.COOKIE, "") == self.token:
            return
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

            # 주소에 ?k= 가 달려 들어왔고 팀 화면이 아니면,
            # 쿠키에 옮겨 담고 깨끗한 주소로 다시 보낸다.
            #   -> 이후로는 주소창에 암호가 보이지 않는다.
            # 팀 화면(--team)은 다른 보드로 건너가야 해서 쿠키가 안 통한다(교차 출처).
            if self.token and not self.boards and request.args.get("k", "") == self.token:
                r = make_response(redirect("/", code=302))
                r.set_cookie(self.COOKIE, self.token,
                             max_age=12 * 3600,     # 12시간
                             httponly=True,         # 자바스크립트가 못 읽는다
                             samesite="Lax")
                return r
            if self.boards:
                import json as _json
                bs = [{"name": n, "base": b.rstrip("/"), "k": k}
                      for n, b in self.boards]
                return TEAM_PAGE.replace("__BOARDS__",
                                         _json.dumps(bs, ensure_ascii=False))
            # 여기까지 왔으면 쿠키가 있다(또는 암호가 없다).
            # 하위 요청(/video_feed 등)은 쿠키로 통과하므로 ?k= 를 안 붙인다.
            return HTML_PAGE.replace("__K__", "")

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
                st["logs"] = list(self._logs)
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
