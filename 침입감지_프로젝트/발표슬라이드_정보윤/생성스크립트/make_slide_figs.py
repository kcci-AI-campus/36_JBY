# -*- coding: utf-8 -*-
# 발표 슬라이드용 그림 (팀장 디자인 팔레트). 모든 그림은 슬라이드에 1:1(인치) 로 놓이므로 figsize 가 곧 슬라이드 위 크기다.
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"C:\Users\LG\Developer\침입감지_프로젝트\발표슬라이드_정보윤"
os.makedirs(OUT, exist_ok=True)
fm.fontManager.addfont(r"C:\Windows\Fonts\malgun.ttf")
fm.fontManager.addfont(r"C:\Windows\Fonts\malgunbd.ttf")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BLUE, NAVY, GRAY, MUTED, LGRAY = "#2563EB", "#0F172A", "#475569", "#64748B", "#94A3B8"
PANEL, WHITE, RED, GREEN, AMBER, ORANGE = "#F1F5F9", "#FFFFFF", "#DC2626", "#16A34A", "#D97706", "#F97316"


def box(ax, x, y, w, h, title, sub="", fc=WHITE, ec=BLUE, tc=NAVY, fs=10, sfs=8, lw=1.8, rs=0.12, tsplit=0.64):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=%s" % rs, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h * (tsplit if sub else 0.5), title, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=sfs, color=GRAY, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=LGRAY, lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=lw, color=color,
                                 connectionstyle="arc3,rad=%s" % rad))


def marker(ax, x, y, n, color=BLUE, r=0.13, fs=8.5):
    ax.add_patch(Circle((x, y), r, fc=color, ec=WHITE, lw=1.2, zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=fs, fontweight="bold", color=WHITE, zorder=6)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=WHITE)
    ax.set_facecolor(WHITE); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=220, facecolor=WHITE); plt.close(fig)
    print("saved", name, Image.open(p).size)


# ───────── 그림 1: 알림 파이프라인 (슬라이드 폭 9.5 in × 2.95 in) ─────────
W, H = 9.5, 2.95
fig, ax = canvas(W, H)
steps = [("① 카메라 → YOLO11n 추론", "USB 웹캠 640×480 · 30 FPS\nbaby · adult · knife · outlet 검출 (LiteRT)"),
         ("② 거리 판정", "위험 반경 = 위험물 박스 폭 × 1.5\n아기 박스가 반경에 닿으면 '위험'"),
         ("③ 체류 판정", "최근 0.5초 프레임의 60 % 이상 위험\n(최소 3장) 이면 경보 → 깜빡임 오검출 제거"),
         ("④ 경보 출력", "보드 LED 즉시 빨강\n휴대폰 알림음 · 웹으로 전달 (아래)")]
bw, bg, y1, bh = 2.2, 0.2, 1.72, 0.98
x = 0.1
for i, (t, s) in enumerate(steps):
    ec = RED if i == 3 else BLUE
    box(ax, x, y1, bw, bh, t, s, ec=ec, tc=(RED if i == 3 else NAVY), fs=9.8, sfs=7.9)
    if i < 3:
        arrow(ax, x + bw, y1 + bh / 2, x + bw + bg, y1 + bh / 2, color=BLUE, lw=1.8)
    x += bw + bg
outs = [("보드 LED", "초록·파랑·빨강 3색, 즉시", GREEN),
        ("휴대폰 알림 (ntfy)", "알림음 + 사진 1장 + 앞 3초·뒤 2초 클립\n같은 경보는 최소 30초 간격", AMBER),
        ("실시간 화면", "알림을 탭하면 웹 페이지\nCloudflare 터널 + 접속 암호", BLUE),
        ("웹 대시보드", "상태·경보 기록·LED·FPS·전력\n(3-4 팀원B 파트)", GRAY)]
ow, og, y2, oh = 2.2, 0.2, 0.2, 0.95
x = 0.1
sx = 0.1 + 3 * (bw + bg) + bw / 2
for i, (t, s, c) in enumerate(outs):
    box(ax, x, y2, ow, oh, t, s, fc=PANEL, ec=c, tc=c, fs=9.6, sfs=7.9, lw=1.6)
    arrow(ax, sx, y1, x + ow / 2, y2 + oh + 0.02, color=RED, lw=1.3)
    x += ow + og
ax.text(0.1, 2.82, "한 프레임의 흐름 — 전부 보드 안에서 처리. 평상시 영상은 밖으로 나가지 않고, 경보 순간의 사진·클립만 나간다",
        fontsize=9.2, color=NAVY, fontweight="bold", va="center")
ax.text(9.4, 1.36, "경보 발생", ha="right", va="center", fontsize=8.2, color=RED, fontweight="bold")
save(fig, "그림_알림파이프라인.png")

# ───────── 그림 2: 게이팅 흐름도 (3.25 × 3.05 in) ─────────
W, H = 3.25, 3.05
fig, ax = canvas(W, H)
box(ax, 0.85, 2.62, 1.55, 0.36, "새 프레임 (카메라)", ec=LGRAY, fs=8.8)
arrow(ax, 1.625, 2.62, 1.625, 2.42, color=LGRAY)
box(ax, 0.55, 1.82, 2.15, 0.58, "직전 프레임과 차이 계산", "160×120 흑백으로 줄여 '변한 픽셀 비율' 만 계산", ec=BLUE, fs=8.8, sfs=6.9)
marker(ax, 0.57, 2.4, "A", color=BLUE, r=0.11, fs=7.5)
arrow(ax, 1.625, 1.82, 1.625, 1.62, color=LGRAY)
ax.add_patch(Polygon([[1.625, 1.62], [2.5, 1.15], [1.625, 0.68], [0.75, 1.15]], closed=True, fc=WHITE, ec=AMBER, lw=1.8))
ax.text(1.625, 1.22, "변한 픽셀 < 0.6 % ?", ha="center", va="center", fontsize=8.6, fontweight="bold", color=NAVY)
ax.text(1.625, 1.02, "그리고 위험 중 아님", ha="center", va="center", fontsize=6.9, color=GRAY)
marker(ax, 0.8, 1.55, "B", color=AMBER, r=0.11, fs=7.5)
arrow(ax, 2.5, 1.15, 2.72, 1.15, color=GREEN)
box(ax, 2.72, 0.8, 0.5, 0.7, "건너뜀", "직전 결과\n재사용", fc="#ECFDF5", ec=GREEN, tc=GREEN, fs=7.6, sfs=6.2, lw=1.4, tsplit=0.7)
ax.text(2.6, 1.28, "예", fontsize=7, color=GREEN, fontweight="bold", ha="center")
arrow(ax, 0.75, 1.15, 0.53, 1.15, color=RED)
box(ax, 0.03, 0.8, 0.5, 0.7, "추론", "YOLO11n\n실행", fc="#FEF2F2", ec=RED, tc=RED, fs=7.6, sfs=6.2, lw=1.4, tsplit=0.7)
ax.text(0.64, 1.28, "아니오", fontsize=7, color=RED, fontweight="bold", ha="center")
ax.add_patch(FancyBboxPatch((0.05, 0.03), 3.15, 0.6, boxstyle="round,pad=0.01,rounding_size=0.06", fc=PANEL, ec="none"))
marker(ax, 0.2, 0.33, "C", color=GRAY, r=0.11, fs=7.5)
ax.text(0.38, 0.33, "안전장치  ① 위험 중이면 절대 건너뛰지 않음\n② 연속 15장 건너뛰면 강제로 한 번 추론\n③ 쉴 때도 8 FPS 로 확인 (최대 공백 0.13초)",
        ha="left", va="center", fontsize=6.9, color=GRAY, linespacing=1.35)
save(fig, "그림_게이팅흐름도.png")

# ───────── 그림 3: 게이팅 Before/After 막대 (두 보드) — 참고용 ─────────
fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=WHITE); ax.set_facecolor(WHITE); ax.set_axisbelow(True)
labels = ["교육장 보드 9/23\n인형·칼 놓인 정지 장면", "지인 보드 9/26\n빈 방, 실제 카메라"]
before, after, pct = [6.03, 7.12], [2.26, 2.52], ["-62 %", "-65 %"]
x, w = np.arange(2), 0.34
ax.bar(x - w / 2, before, w, color=LGRAY, label="게이팅 끔", lw=0)
ax.bar(x + w / 2, after, w, color=BLUE, label="게이팅 켬", lw=0)
for xi, v in zip(x - w / 2, before): ax.text(xi, v + 0.12, "%.2f W" % v, ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
for xi, v in zip(x + w / 2, after): ax.text(xi, v + 0.12, "%.2f W" % v, ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
for xi, b, p in zip(x, before, pct): ax.text(xi, b + 0.8, p, ha="center", fontsize=12, color=RED, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, color=NAVY); ax.set_ylim(0, 9)
ax.set_ylabel("평균 전력 (W, PMIC, 웹캠 제외)", color=GRAY, fontsize=9)
ax.grid(axis="y", color="#E2E8F0"); ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#E2E8F0"); ax.tick_params(colors=GRAY, labelsize=8.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "그림_전력_BeforeAfter.png"), dpi=200); plt.close(fig)

# ───────── 그림 4: 4상태 상태도 (6.25 × 2.1 in) ─────────
W, H = 6.25, 2.1
fig, ax = canvas(W, H)
states = [("① 대기", "아기 없음\n5초에 1장 · 클럭 1500", BLUE),
          ("② 감시", "아기 있음, 위험물 멀리\n5 FPS · 게이팅 · 클럭 1500", GREEN),
          ("③ 경계", "거리 < 반경 × 2\n15 FPS · 클럭 2400", AMBER),
          ("④ 경보", "반경 안 0.5초 체류\n30 FPS · LED · 휴대폰 알림", RED)]
bw, bh, bg, y0 = 1.32, 0.95, 0.25, 0.62
for i, (t, s, c) in enumerate(states):
    bx = 0.1 + i * (bw + bg)
    box(ax, bx, y0, bw, bh, t, s, ec=c, tc=c, fs=10.5, sfs=7.4, lw=1.9, rs=0.1)
ups = ["아기 보임", "위험물에 접근", "반경 안 0.5초"]
downs = ["아기 3초 안 보임", "멀어진 지 3초", "위험 없어진 지 3초"]
for i in range(3):
    x1 = 0.1 + i * (bw + bg) + bw; x2 = x1 + bg
    c = states[i + 1][2]
    arrow(ax, x1, y0 + bh * 0.7, x2, y0 + bh * 0.7, color=c, lw=1.6)
    ax.text((x1 + x2) / 2, y0 + bh + 0.1, ups[i], ha="center", fontsize=7.2, color=c, fontweight="bold")
    arrow(ax, x2, y0 + bh * 0.3, x1, y0 + bh * 0.3, color=LGRAY, lw=1.3)
    ax.text((x1 + x2) / 2, y0 - 0.12, downs[i], ha="center", fontsize=6.8, color=MUTED)
ax.text(W / 2, 1.9, "상태마다 '얼마나 자주 · 어떤 클럭으로' 추론할지를 다르게 — 경계 구역(반경×2)이 완충 역할",
        ha="center", va="center", fontsize=8.6, color=GRAY)
ax.text(W / 2, 0.22, "원리: 전력 = 유휴 + 추론 1회 에너지 × 초당 횟수  →  위험이 멀수록 천천히·싸게, 가까울수록 빠르게·정확하게",
        ha="center", va="center", fontsize=8.4, color=NAVY, fontweight="bold")
save(fig, "그림_4상태_상태도.png")

# ───────── 그림 5: 두 장면 × 3설계 막대 (3.6 × 1.7 in) ─────────
fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.7), facecolor=WHITE)
fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.2, wspace=0.12)
designs = [("두 모드", LGRAY), ("4상태", BLUE), ("4상태+INT8+클럭", ORANGE)]
data = [("아기 움직임, 위험물 없음", [4.29, 2.53, 1.66]), ("아기+칼 반경 안 (경보 구간)", [7.17, 6.75, 2.47])]
for ax, (title, vals) in zip(axes, data):
    ax.set_facecolor(WHITE); ax.set_axisbelow(True)
    for j, (v, (n, c)) in enumerate(zip(vals, designs)):
        ax.bar(j, v, 0.62, color=c, lw=0)
        ax.text(j, v + 0.15, "%.1f" % v, ha="center", fontsize=7.6, color=NAVY, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([d[0] for d in designs], fontsize=5.9, color=GRAY)
    ax.set_yticks([]); ax.set_ylim(0, 8.6)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.set_title(title, fontsize=7.6, color=NAVY, fontweight="bold", pad=3)
axes[0].text(-0.45, 8.1, "W", fontsize=6.5, color=MUTED)
save(fig, "그림_장면별전력_2장면.png")

# ───────── 웹 대시보드 캡처: 브라우저 테두리 잘라내고 위쪽(헤더+상태 카드)만 ─────────
src = os.path.join(OUT, "웹대시보드_캡처.jpg")
im = Image.open(src).convert("RGB")
arr = np.asarray(im).mean(axis=2)
rows_dark = np.where(arr.mean(axis=1) < 60)[0]
y0 = int(rows_dark[0]) if len(rows_dark) else 0
wpx = im.size[0]
crop = im.crop((0, y0, wpx, min(im.size[1], y0 + int(0.33 * wpx))))
crop.save(os.path.join(OUT, "웹대시보드_상단.png"))
print("dashboard crop", y0, crop.size)
print("figures done:", sorted(f for f in os.listdir(OUT) if f.startswith("그림_")))
