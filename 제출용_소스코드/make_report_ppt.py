# -*- coding: utf-8 -*-
"""
결과보고서 PPT 생성기.

교수님이 주신 양식(최종보고서(포맷예시)(포맷자유).pptx)의 6개 장 구성을 따르고,
오늘까지 실측한 수치와 게임 캡처를 채워 넣는다.

실행:
    Lec_python\\집에서_미리해보기\\.venv\\Scripts\\python.exe make_report_ppt.py

만들어지는 것:
    AI_시스템반도체_설계_2기_결과보고서(정보윤).pptx
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

NAME = '정보윤'
OUT = 'AI_시스템반도체_설계_2기_결과보고서(%s).pptx' % NAME

SHOT = r'C:\Users\kccistc\Pictures\Screenshots\가위바위보'
BASE = r'C:\Users\kccistc\Developer'
IMG_GAME = os.path.join(SHOT, 'v3', '스크린샷 2026-09-17 173242.png')
IMG_V3TERM = os.path.join(SHOT, 'v3', '스크린샷 2026-09-17 173724.png')
IMG_V1TERM = os.path.join(SHOT, 'v1', '스크린샷 2026-09-17 173714.png')
IMG_LABEL = r'C:\Users\kccistc\Developer\라벨확인_확대비교.png'
IMG_GAME_LBL = r'C:\Users\kccistc\Developer\게임화면_라벨.png'
IMG_LABELVS = os.path.join(BASE, '라벨비교_두사람.png')

# 색
NAVY = RGBColor(0x1F, 0x3B, 0x63)
BLUE = RGBColor(0x2E, 0x6D, 0xB4)
GRAY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x8A, 0x4C)
LIGHT = RGBColor(0xEE, 0xF3, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = '맑은 고딕'

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

W = prs.slide_width
H = prs.slide_height


def setfont(run, size=14, bold=False, color=None, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    # 한글 글꼴을 동아시아 글꼴로도 지정해야 안 깨진다
    rPr = run._r.get_or_add_rPr()
    from pptx.oxml.ns import qn
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        rPr.append(ea)
    ea.set('typeface', font)


def textbox(slide, x, y, w, h, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = align
    return tf


def para(tf, text, size=14, bold=False, color=None, space_before=4, first=False,
         align=PP_ALIGN.LEFT, indent=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.level = indent
    r = p.add_run()
    r.text = text
    setfont(r, size, bold, color)
    return p


def slide_header(slide, num, title, sub=None):
    """상단 제목 띠."""
    bar = slide.shapes.add_shape(1, 0, 0, W, Inches(1.0))   # 1 = 사각형
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.45)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = ('%s. %s' % (num, title)) if num else title
    setfont(r, 26, True, WHITE)
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        setfont(r2, 13, False, RGBColor(0xC8, 0xD8, 0xEE))


def conclusion_box(slide, x, y, w, text, color=BLUE, size=15):
    """두괄식 결론 상자."""
    box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(0.62))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = color
    box.line.width = Pt(1.5)
    tf = box.text_frame
    tf.margin_left = Inches(0.2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = text
    setfont(r, size, True, NAVY)
    return box


def table(slide, x, y, w, rows, col_w=None, size=12, head_size=12,
          highlight_rows=(), highlight_col=None, row_h=0.32):
    """rows[0] 이 머리글. highlight_rows 는 강조할 행 번호(0부터)."""
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * nr))
    tbl = shp.table
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            if i == 0:
                setfont(r, head_size, True, WHITE)
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            else:
                emph = (i in highlight_rows) or (highlight_col is not None and j == highlight_col)
                setfont(r, size, emph, RED if emph else GRAY)
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT if (i in highlight_rows) else WHITE
    return tbl


def note(slide, x, y, w, text, size=11, color=GRAY):
    tf = textbox(slide, x, y, w, 0.5)
    para(tf, text, size, False, color, first=True)


def picture(slide, path, x, y, h=None, w=None):
    if not os.path.exists(path):
        ph = slide.shapes.add_shape(1, Inches(x), Inches(y),
                                    Inches(w or 4), Inches(h or 2.5))
        ph.fill.solid()
        ph.fill.fore_color.rgb = LIGHT
        ph.line.color.rgb = GRAY
        r = ph.text_frame.paragraphs[0].add_run()
        r.text = '[그림 자리]\n' + os.path.basename(path)
        setfont(r, 11, False, GRAY)
        return ph
    kw = {}
    if h:
        kw['height'] = Inches(h)
    if w:
        kw['width'] = Inches(w)
    return slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def caption(slide, x, y, w, text):
    tf = textbox(slide, x, y, w, 0.3, PP_ALIGN.CENTER)
    para(tf, text, 10, True, BLUE, first=True, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════
# 표지
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()

tf = textbox(s, 1.0, 2.1, 11.3, 1.0)
para(tf, 'AI 시스템반도체 SW개발자 (2기)', 18, False,
     RGBColor(0xA9, 0xC4, 0xE4), first=True)
tf = textbox(s, 1.0, 2.7, 11.3, 1.6)
para(tf, '라즈베리파이 5 + YOLO11n 기반', 30, True, WHITE, first=True)
para(tf, '2인용 가위바위보 게임 On-Device 구현', 40, True, WHITE)

line = s.shapes.add_shape(1, Inches(1.0), Inches(4.75), Inches(3.2), Pt(3))
line.fill.solid()
line.fill.fore_color.rgb = RGBColor(0x5B, 0x9B, 0xD5)
line.line.fill.background()

tf = textbox(s, 1.0, 5.0, 11.3, 1.4)
para(tf, '프로젝트 결과보고서', 20, True, RGBColor(0xC8, 0xD8, 0xEE), first=True)
para(tf, NAME, 18, False, WHITE, space_before=10)
para(tf, '2026. 09. 18.', 14, False, RGBColor(0xA9, 0xC4, 0xE4))

# ═══════════════════════════════════════════════════════════
# 목차
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, None, '목차')
items = [
    ('1', '주제 및 결과 요약', '무엇을 만들었고 결과가 얼마인가'),
    ('2', '개발 목표 및 개발 결과', '5단계 과제 구현 + 추가로 한 것'),
    ('3', '핵심 기술', '검출 파이프라인 · 모델 경량화 · 데이터셋 도구'),
    ('4', '결과 분석 및 기대 효과', '두 개의 시험지로 드러난 실제 성능'),
    ('5', '향후 연구 과제', '남은 문제와 해결 방향'),
    ('6', '프로젝트 수행 후기', '측정으로 배운 것'),
]
y = 1.5
for n, t, d in items:
    box = s.shapes.add_shape(1, Inches(1.2), Inches(y), Inches(0.62), Inches(0.62))
    box.fill.solid()
    box.fill.fore_color.rgb = BLUE
    box.line.fill.background()
    r = box.text_frame.paragraphs[0].add_run()
    r.text = n
    setfont(r, 20, True, WHITE)
    box.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    tf = textbox(s, 2.1, y - 0.02, 9.5, 0.7)
    para(tf, t, 19, True, NAVY, first=True)
    para(tf, d, 12, False, GRAY, space_before=1)
    y += 0.92

# ═══════════════════════════════════════════════════════════
# 1. 주제 및 결과 요약
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '1', '주제 및 결과 요약')

conclusion_box(s, 0.5, 1.18, 12.3,
               '학습 데이터 104장 → 724장 확대로 실사용 검출 성능 mAP50 0.511 → 0.853 (+67%) 달성',
               size=17)

# 큰 숫자 타일 4개 (레퍼런스 7번)
tiles = [('0.853', '실사용 mAP50', 'v1 0.511 → +67%'),
         ('9 / 9', '보드 판정 정확도', 'v1 3/9'),
         ('68.1', 'FPS', 'INT8 · MJPG · 4스레드'),
         ('2.9 MB', '모델 크기', 'float32 10.1 MB → 3.5배↓')]
x = 0.5
for big, t, sub in tiles:
    box = s.shapes.add_shape(1, Inches(x), Inches(2.0), Inches(2.95), Inches(1.5))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = BLUE; box.line.width = Pt(1.5)
    tf = box.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p0 = tf.paragraphs[0]; p0.alignment = PP_ALIGN.CENTER
    r = p0.add_run(); r.text = big; setfont(r, 34, True, NAVY)
    p1 = tf.add_paragraph(); p1.alignment = PP_ALIGN.CENTER
    r = p1.add_run(); r.text = t; setfont(r, 13, True, BLUE)
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r = p2.add_run(); r.text = sub; setfont(r, 10.5, False, GRAY)
    x += 3.1

# Overview / Goal / Conclusion 카드 (레퍼런스 12번) — 왼쪽 세로 배치
cards = [('Overview',
          ['라즈베리파이 5 단독 동작 2인용 가위바위보 게임',
           'YOLO11n으로 두 손 검출 → 좌/우 구분 → 승패 판정']),
         ('Goal',
          ['① 5단계 과제 전부 구현   ② 보드 단독 실시간 동작',
           '③ 실사용 환경 인식 정확도 확보']),
         ('Conclusion',
          ['5단계 전부 구현 · 판정 9/9 · 68.1 FPS',
           '시험지 2분할로 실사용 성능 확인 → 데이터 확대로 +79%'])]
y = 3.45
for t, lines_ in cards:
    box = s.shapes.add_shape(1, Inches(0.5), Inches(y), Inches(6.0), Inches(1.05))
    box.fill.solid(); box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = NAVY; box.line.width = Pt(1.2)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.06)
    r = tf.paragraphs[0].add_run(); r.text = t; setfont(r, 14, True, NAVY)
    for ln in lines_:
        pp = tf.add_paragraph()
        r = pp.add_run(); r.text = ln; setfont(r, 11, False, GRAY)
    y += 1.15

# 그림은 오른쪽. 960x540 이므로 폭 6.0in -> 높이 3.38in (3.45 + 3.38 = 6.83)
picture(s, IMG_GAME_LBL, 6.85, 3.45, w=6.0)
tf = textbox(s, 6.85, 6.88, 6.0, 0.55)
para(tf, '▲ 그림 1. 게임 실행 화면', 10.5, True, BLUE, first=True)
para(tf, '① 스코어보드   ② FPS   ③ P1 박스(보)   ④ P2 박스(바위)   ⑤ 판정 · 프레임 고정',
     10, False, GRAY, space_before=2)

# ═══════════════════════════════════════════════════════════
# 2-1. 5단계 과제
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '2', '개발 목표 및 개발 결과', '2-1. 교수님 제시 5단계 과제')

conclusion_box(s, 0.5, 1.25, 12.3, '5단계 전 과제 구현 완료 — 9가지 조합 보드 실측 판정 정확도 9/9', size=16)

rows = [
    ['단계', '구현 과제', '구현', '근거 (RPS_Game.py)'],
    ['1', 'Bounding Box X좌표로 P1(좌)/P2(우) 구분', '완료', 'split_players() — 상자 중심 x 정렬'],
    ['2', 'P1 vs P2 클래스 비교 → 승/무/패 출력', '완료', 'BEATS 딕셔너리 + judge()'],
    ['3', '손 2개가 아닐 때 예외 메시지', '완료', 'len(boxes) 분기 → 화면 하단 안내'],
    ['4', '3-2-1 카운트다운 후 판정 순간 Freeze', '완료', '상태 기계 + frame.copy()'],
    ['5', '스코어보드 · 승패 효과음 · 재시작 키', '완료', 'draw_scoreboard() / play() / SPACE·r·q'],
]
table(s, 0.5, 2.1, 12.3, rows, col_w=[0.7, 4.6, 0.9, 5.1], size=11.5, row_h=0.45)

tf = textbox(s, 0.5, 4.85, 12.3, 2.3)
para(tf, '4단계 상태 기계 — 실패하던 판정의 수정', 15, True, NAVY, first=True)
para(tf, 'READY ─SPACE→ COUNT(3초) ─0초→ JUDGING(유예 2초) ─손 2개→ RESULT(2.5초) → READY',
     14, True, BLUE, space_before=8)
para(tf, '· 문제 : 카운트 0인 "한 프레임"에서만 판정 → 11 FPS 환경에서 대부분 취소', 13, space_before=8)
para(tf, '· 원인 : 손을 내미는 동작 중 검출이 흔들려 그 순간 두 손이 잡힐 확률 낮음', 13)
para(tf, '· 해결 : JUDGING 유예 구간 2초 도입 → "0 이후 두 손이 처음 잡히는 프레임"으로 판정', 13)
para(tf, '· 결과 : 판정 취소 해소, 유예 안에 못 찾으면 Cancelled 안내 후 복귀 (3단계와 연결)',
     13, True, GREEN)

# ═══════════════════════════════════════════════════════════
# 2-2. 과제 밖으로 더 한 것
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '2', '개발 목표 및 개발 결과', '2-2. 과제 범위 밖에서 발견하고 해결한 것')

conclusion_box(s, 0.5, 1.25, 12.3,
               '"동작 확인"에서 멈추지 않고 측정 → 원인 규명 → 도구 제작 → 재검증 순으로 진행', size=15)

rows = [
    ['발견한 문제', '왜 문제인가', '만든 것 / 결과'],
    ['느린데 어디가 느린지 모름', '인상만으로는 개선 불가',
     'trace_util.py — Chrome Trace 타임라인 직접 구현'],
    ['한 프레임 60.5 ms 중 추론 밖 70%', '모델 경량화만으로는 개선 없음',
     '파이프라인 4단계 최적화 → 60.54 → 23.19 ms'],
    ['mAP 0.995인데 어두운 옷 앞 실패', '검증 점수가 실사용 성능과 불일치',
     '학습 데이터 104장 픽셀 직접 측정 → 원인 규명'],
    ['교수님 데이터가 xml이라 재학습 불가', 'YOLO는 txt만 인식, 형식 오류는 무음 실패',
     'make_dataset.py — 변환·병합·전수 검사'],
    ['라벨 실수가 무음으로 학습을 훼손', '클래스 번호가 밀려도 오류 없음',
     'check_labels.py — 폴더명과 라벨 기계 대조'],
    ['혼자서 데이터 촬영 불가', '키 60회 입력하며 손 제시 불가능',
     'EX_01_Auto_Capture.py — 자동 촬영 + 중복 제거'],
]
table(s, 0.5, 2.1, 12.3, rows, col_w=[3.3, 3.9, 5.1], size=11, row_h=0.52)

note(s, 0.5, 6.3, 12.3,
     '· 직접 작성 파일 10종 제출물 포함    · 교수님 원본 예제는 수정하지 않고 전부 사본으로 작업')

# ═══════════════════════════════════════════════════════════
# 3-1. 검출 파이프라인
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '3', '핵심 기술', '3-1. 분류에서 검출로 — 게임이 성립한 이유')

conclusion_box(s, 0.5, 1.25, 12.3,
               '검출 모델은 "무엇"과 "어디"를 함께 출력 → 두 손 동시 처리 가능', size=16)

rows = [
    ['', '이전 실습 (MobileNetV2 분류)', '본 프로젝트 (YOLO11n 검출)'],
    ['모델 입력', 'MediaPipe가 잘라낸 손 조각', '사진 전체 320×320'],
    ['출력', '클래스 1개', '박스 후보 2,100개 → 필터 → 최종 박스'],
    ['손 2개 처리', '불가', '가능 — 게임 성립의 전제'],
    ['손 위치', '알 수 없음', 'x 좌표로 P1/P2 구분'],
    ['후처리', '없음 (argmax 1줄)', '약 40줄 (필터 → NMS → 좌표 복원)'],
]
table(s, 0.5, 2.1, 12.3, rows, col_w=[1.9, 5.0, 5.4], size=11.5,
      highlight_rows=(3,), row_h=0.42)

tf = textbox(s, 0.5, 4.85, 12.3, 2.2)
para(tf, '직접 구현한 후처리 3단계', 15, True, NAVY, first=True)
para(tf, '① 신뢰도 필터 — 후보 2,100개 중 conf ≥ 0.4 만 통과', 13, space_before=7)
para(tf, '② NMS — 같은 손의 중복 박스 정리 (cv2.dnn.NMSBoxesBatched, IoU 0.45)', 13)
para(tf, '③ 좌표 복원 — letterbox(비율 유지 + 회색 114 여백) 좌표를 원본 해상도로 역변환', 13)
para(tf, '· 측정 결과 : 후처리 전체 0.36 ms/프레임 → 코드 길이와 실행 시간은 무관',
     13, True, GREEN, space_before=8)

# ═══════════════════════════════════════════════════════════
# 3-2. 경량화 + 파이프라인
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '3', '핵심 기술', '3-2. 모델 경량화(PTQ)와 파이프라인 최적화')

conclusion_box(s, 0.5, 1.22, 6.0, '모델 3.5배 축소 · 정확도 손실 없음', size=14)
conclusion_box(s, 6.8, 1.22, 6.0, '한 프레임 60.54 → 23.19 ms (2.6배)', size=14)

tf = textbox(s, 0.5, 1.98, 6.0, 0.35)
para(tf, 'PTQ 양자화 결과', 14, True, NAVY, first=True)
rows = [
    ['모델', '크기', '방식'],
    ['best.tflite', '10.1 MB', 'float32 (기준)'],
    ['best_int8.tflite', '2.9 MB', 'PTQ INT8 — 보정 데이터 필요'],
    ['best_w8a32.tflite', '2.8 MB', 'Dynamic Range — 보정 불필요'],
]
table(s, 0.5, 2.38, 6.0, rows, col_w=[2.2, 1.3, 2.5], size=11, row_h=0.38)
note(s, 0.5, 3.98, 6.0,
     '· 세 모델 모두 입출력 float32 유지 → 보드 코드에서 scale/zero 환산 없이 파일명만 교체\n'
     '· 프루닝·클러스터링 미적용 — 04 실습 측정 결과 tflite 크기·추론 시간 변화 없음(압축 크기만 감소)')

tf = textbox(s, 6.8, 1.98, 6.0, 0.35)
para(tf, '파이프라인 4단계 최적화', 14, True, NAVY, first=True)
rows = [
    ['단계', '조치', '한 프레임'],
    ['기준', '원본 코드', '60.54 ms'],
    ['1', 'waitKey(10) → waitKey(1)', '48.3 ms'],
    ['2', '카메라 읽기 별도 스레드', '34.9 ms'],
    ['3', '검출 주기 2프레임', '27.1 ms'],
    ['4', '인터프리터 스레드 4개', '23.19 ms'],
]
table(s, 6.8, 2.38, 6.0, rows, col_w=[0.9, 3.4, 1.7], size=11,
      highlight_rows=(5,), row_h=0.38)

tf = textbox(s, 0.5, 4.95, 12.3, 2.2)
para(tf, '측정이 뒤집은 예측 — 추측으로는 알 수 없던 것', 15, True, NAVY, first=True)
para(tf, '· cv2.imshow 병목 예상 → 실측 0.10 ms. 실제 병목은 waitKey 11.98 ms '
         '(imshow는 비동기, 화면 전송이 waitKey 내부에서 발생)', 13, space_before=7)
para(tf, '· 후처리 40줄 부담 예상 → 실측 0.36 ms. 추론이 전체의 84% 차지', 13)
para(tf, '· 카메라 형식 MJPG 전환 → 초당 처리 장수 상한 18.5장 해소, 실측 68.1 FPS', 13)
para(tf, '· 결론 : 성능 개선은 추측이 아니라 측정에서 출발', 13, True, GREEN, space_before=7)

# ═══════════════════════════════════════════════════════════
# 3-3. 데이터셋 도구
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '3', '핵심 기술', '3-3. 데이터셋 구축 자동화와 전수 검증')

conclusion_box(s, 0.5, 1.25, 12.3,
               '라벨 오류는 오류 메시지 없이 학습을 훼손 → "사람이 확인"이 아니라 "기계가 대조"하도록 구현',
               size=15)

tf = textbox(s, 0.5, 2.0, 6.0, 2.6)
para(tf, '문제 — 같은 이름의 데이터셋 zip 두 개', 14, True, NAVY, first=True)
para(tf, '· A : 라벨 txt — YOLO 정식 구조, 학습 가능', 12.5, space_before=6)
para(tf, '· B : 라벨 xml — LabelImg 원본, 학습 불가', 12.5)
para(tf, '· LabelImg 기본 저장 형식이 xml → 새로 라벨링하면 학습 불가 쪽 생성', 12.5)
para(tf, '· Ultralytics는 라벨 0개로 인식해도 오류 없이 학습 진행', 12.5, True, RED)
para(tf, '해결 — make_dataset.py', 14, True, NAVY, space_before=12)
para(tf, '· xml→txt 변환 · 팀 라벨 병합 · train/test 분할 · data.yaml 생성 · 전수 검사 · zip 묶기',
     12.5, space_before=5)
para(tf, '· 해시 대조로 중복 사진 287장 자동 제외', 12.5)

tf = textbox(s, 6.8, 2.0, 6.0, 0.35)
para(tf, '변환기 검증 — 교수님 정답 txt와 1:1 대조', 14, True, NAVY, first=True)
rows = [
    ['대조 항목', '결과'],
    ['대조 박스 수', '132개'],
    ['클래스 번호 불일치', '0개'],
    ['좌표 최대 차이', '0.00064 픽셀'],
]
table(s, 6.8, 2.4, 6.0, rows, col_w=[3.6, 2.4], size=11.5,
      highlight_rows=(3,), row_h=0.38)
note(s, 6.8, 4.0, 6.0, '· 차이는 전부 소수점 6자리 반올림 방식 차이')

tf = textbox(s, 6.8, 4.5, 6.0, 2.4)
para(tf, 'check_labels.py — 라벨 전수 검사', 14, True, NAVY, first=True)
para(tf, '· 원리 : 폴더 이름이 곧 정답 (paper_rock → 보 1 + 바위 1)', 12.5, space_before=6)
para(tf, '· 실제 검출 오류 : 오타 클래스, 파일명 손상 4건, 클래스 오기 2건, 사진 없는 라벨', 12.5)
para(tf, '· 팀원 3명 라벨 306개 전수 대조 후 병합 (불일치 0건 확인)', 12.5, True, GREEN)

# ═══════════════════════════════════════════════════════════
# 4-1. 핵심 표
# ═══════════════════════════════════════════════════════════
# 4-0. 결과 분석 한 장 요약
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '4', '결과 분석 및 기대 효과', '한 장 요약 — 상세 근거는 다음 6장')

conclusion_box(s, 0.5, 1.2, 12.3,
               '5단계 과제 전부 구현 · 보드 단독 68.1 FPS · 실사용 인식 성능 mAP50 0.511 → 0.853 (+67%)',
               size=16)

tf = textbox(s, 0.5, 1.95, 6.1, 0.35)
para(tf, '① 목표 달성 여부', 14, True, NAVY, first=True)
rows = [
    ['목표', '기준', '결과', '달성'],
    ['5단계 과제', '전부', '전부 구현', 'O'],
    ['보드 단독 실시간', '10 FPS 이상', '68.1 FPS', 'O'],
    ['승패 판정', '9가지 조합', '9 / 9', 'O'],
    ['실사용 인식 성능', '—', 'mAP50 0.853', 'O'],
]
table(s, 0.5, 2.35, 6.1, rows, col_w=[1.8, 1.5, 1.7, 1.1], size=11,
      highlight_rows=(4,), row_h=0.33)

tf = textbox(s, 0.5, 4.2, 6.1, 0.35)
para(tf, '③ 기대 효과', 14, True, NAVY, first=True)
tf = textbox(s, 0.5, 4.58, 6.1, 2.5)
para(tf, '· 모델 2.9 MB / 68.1 FPS — 라즈베리파이 5 단독으로 서버 없이 동작',
     12, first=True)
para(tf, '· 데이터셋 도구(변환·병합·전수검증)는 클래스 이름만 바꾸면 다른 검출 과제에 재사용 가능',
     12)
para(tf, '· "시험지를 분리해 측정한다"는 절차 자체가 다음 과제에 그대로 적용됨',
     12, True, GREEN)
para(tf, '· 문턱값 최적화는 재학습 없이 적용 가능해, 현장 조건이 바뀌어도 즉시 대응',
     12)

tf = textbox(s, 6.85, 1.95, 6.1, 0.35)
para(tf, '② 성능이 낮았던 부분 — 원인과 개선책', 14, True, NAVY, first=True)
rows = [
    ['낮았던 것', '원인 (측정값)', '개선책'],
    ['실사용 인식률\n(v1 mAP50 0.511)',
     '학습·시험 조건이 동일해\n실사용 분포를 못 담음',
     '조건별 촬영으로 데이터 확대\n→ 0.853 (달성)'],
    ['게임 판정률\n(자세의 25.8%)',
     'CONF_TH 0.4 가 낮은 신뢰도의\n정답까지 잘라 냄',
     '문턱값 0.10 · 상위 2개 사용\n→ 80.8% (달성)'],
    ['박스 정밀도\n(mAP50-95 0.448)',
     '담당자별 박스 기준 상이\nIoU 0.693, 보 면적 2배',
     '기준 문서화 후 재라벨링\nA/B 실험 설계 완료'],
]
table(s, 6.85, 2.35, 6.1, rows, col_w=[1.8, 2.2, 2.1], size=9.5, row_h=0.78)

note(s, 6.85, 5.5, 6.1,
     '· 세 항목 모두 원인을 수치로 특정했고, 둘은 이미 개선을 적용해 결과를 확인함\n'
     '· 상세 근거: 4-1 두 시험지 비교 / 4-2 결론 철회 / 4-5 라벨 품질 / 4-6 문턱값')

# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '4', '결과 분석 및 기대 효과',
             '4-1. 같은 모델을 두 시험지로 채점 — 결론이 뒤집힌다')

conclusion_box(s, 0.5, 1.2, 12.3,
               '교수님 시험지로는 세 판이 모두 mAP50 0.995로 구분 불가 — 실사용 시험지에서만 0.511 → 0.853 차이가 드러남',
               size=15)

rows = [
    ['모델', '학습 장수', '교수님 22장 (박스 32)', '오늘 233장 (박스 328)'],
    ['v1', '104장 (교수님 원본)', '0.995 / 0.810', '0.511 / 0.339'],
    ['v3', '724장', '0.995 / 0.763', '0.853 / 0.448'],
    ['v9', '1,318장', '0.995 / 0.836', '0.643 / 0.361'],
]
table(s, 0.5, 2.0, 12.3, rows, col_w=[1.2, 3.1, 4.0, 4.0], size=13,
      head_size=12, highlight_rows=(2,), row_h=0.46)
note(s, 0.5, 4.1, 12.3,
     '각 칸 = mAP50 / mAP50-95     · 학습 설정 고정, 데이터만 변경 (YOLO11n, 80 epoch, imgsz 320)\n'
     '· 오늘 시험지 233장은 2026-09-18 직접 촬영·라벨링. 세 판 모두 학습에 사용하지 않음')

tf = textbox(s, 0.5, 4.95, 6.0, 2.3)
para(tf, '결론', 15, True, NAVY, first=True)
para(tf, '· 쉬운 시험지는 학습 데이터를 12.7배 늘린 차이를 전혀 못 보여 줌', 12.5, space_before=6)
para(tf, '· v1은 실사용 조건에서 0.511 — 교수님 시험지로는 0.995', 12.5)
para(tf, '· 시험지를 바꾸자 v3와 v9의 순위가 실제로 뒤집힘', 12.5, True, GREEN)

tf = textbox(s, 6.8, 4.95, 6.0, 2.3)
para(tf, '고찰', 15, True, NAVY, first=True)
para(tf, '· 박스 32개 시험은 epoch 간 변동폭(0.08)보다 차이가 작아 판독 불가', 12.5, space_before=6)
para(tf, '· 328박스 시험에서만 유효한 비교가 성립', 12.5)
para(tf, '· "mAP 0.995 달성"은 조건 없이는 정보가 아님', 12.5, True, RED)

# ═══════════════════════════════════════════════════════════
# 4-2. v4 실험
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '4', '결과 분석 및 기대 효과',
             '4-2. 잘못된 비교를 발견하고 결론을 철회한 과정')

conclusion_box(s, 0.5, 1.22, 12.3,
               '"데이터를 늘렸더니 나빠졌다"로 진단했으나, 실제 원인은 학습이 2 epoch에서 중단된 것 — 결론 철회',
               size=15)

# ── 왼쪽 : 현상과 추적 ──────────────────────────────────────
tf = textbox(s, 0.5, 1.98, 6.1, 0.35)
para(tf, '① 현상 — 앞뒤가 맞지 않는 수치', 14, True, NAVY, first=True)

tf = textbox(s, 0.5, 2.36, 6.1, 1.0)
para(tf, '· v4는 교수님 22장을 val로 써서 best epoch를 선택한 모델', 12, first=True)
para(tf, '· 그런데 바로 그 22장에서 mAP50 0.632', 12, True, RED)
para(tf, '· 자기가 고른 시험지에서 나올 수 없는 점수', 12)

tf = textbox(s, 0.5, 3.45, 6.1, 0.35)
para(tf, '② 추적 — 학습 기록 대조', 14, True, NAVY, first=True)
rows = [
    ['학습 폴더', '학습 데이터', '실제 epoch'],
    ['rps_yolo11n-2', 'v2', '80'],
    ['rps_yolo11n-4', 'v3', '80'],
    ['rps_yolo11n', 'v4', '2'],
]
table(s, 0.5, 3.85, 6.1, rows, col_w=[2.4, 1.8, 1.9], size=11.5,
      highlight_rows=(3,), row_h=0.36)
note(s, 0.5, 5.35, 6.1,
     '· 폴더 이름의 -2, -4 는 ultralytics 실행 순번일 뿐 판 번호와 무관\n'
     '· args.yaml 의 data 줄로 식별해야 정확')

tf = textbox(s, 0.5, 6.05, 6.1, 1.1)
para(tf, '③ 증거 — 두 시험지에서 완전 일치', 14, True, NAVY, first=True)
para(tf, '· save/v4/best.tflite 와 2 epoch 모델의 mAP50 / mAP50-95 /', 11.5, space_before=4)
para(tf, '  Precision / Recall 이 소수점 셋째 자리까지 동일 → 같은 파일', 11.5, True, RED)

# ── 오른쪽 : 무엇이 바뀌었나 ────────────────────────────────
tf = textbox(s, 6.75, 1.98, 6.1, 0.35)
para(tf, '④ 철회되는 주장', 14, True, NAVY, first=True)

tf = textbox(s, 6.75, 2.36, 6.1, 1.1)
para(tf, '· "데이터 47% 증가로 성능 하락"  → 근거 없음', 12, True, RED, first=True)
para(tf, '· "팀원 라벨 기준 차이가 원인"    → 근거 없음', 12, True, RED)
para(tf, '· v4 데이터는 한 번도 정상 평가된 적 없음', 12)

tf = textbox(s, 6.75, 3.45, 6.1, 0.35)
para(tf, '⑤ 조치 — 재발 방지 코드', 14, True, NAVY, first=True)

rows = [
    ['적용', '내용'],
    ['모델 선택', 'args.yaml 의 data 줄로 판 식별'],
    ['중단 감지', 'epoch < 80 이면 RuntimeError'],
    ['기록 보존', '학습 폴더를 Drive 에 복사'],
]
table(s, 6.75, 3.85, 6.1, rows, col_w=[1.7, 4.4], size=11.5,
      highlight_rows=(2,), row_h=0.36)

tf = textbox(s, 6.75, 5.35, 6.1, 1.9)
para(tf, '고찰', 15, True, NAVY, first=True)
para(tf, '· 비교 실험은 "데이터 외 조건이 같은가"를 먼저 확인해야 함', 12, space_before=6)
para(tf, '  — 실제로는 학습 시간이 40배 달랐음', 11.5, False, GRAY)
para(tf, '· 결론에 맞지 않는 수치를 만나면 결론이 아니라 측정을 의심', 12, True, GREEN)
para(tf, '· 자동 생성된 이름(-2, -4)을 판 번호로 추측한 것이 화근', 12)

# ═══════════════════════════════════════════════════════════
# 4-3. 바위 원인 분석
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '4', '결과 분석 및 기대 효과', '4-3. "바위를 못 잡는다"의 원인 분석')

conclusion_box(s, 0.5, 1.22, 12.3, '부족했던 것은 데이터의 "개수"가 아니라 "크기의 다양성"', size=16)

tf = textbox(s, 0.5, 1.98, 6.1, 0.35)
para(tf, '① 현상 — 보드 실측 (9가지 조합 각 1회)', 13.5, True, NAVY, first=True)
rows = [
    ['모델', 'P1 승', '무승부', 'P2 승', '판정 정확도'],
    ['정답', '3', '3', '3', '—'],
    ['v1', '2', '6', '1', '3 / 9'],
    ['v3', '3', '3', '3', '9 / 9'],
]
table(s, 0.5, 2.38, 6.1, rows, col_w=[1.3, 1.1, 1.2, 1.1, 1.4], size=11,
      highlight_rows=(3,), row_h=0.36)
note(s, 0.5, 3.85, 6.1, '· v1은 바위를 오인식해 "거짓 무승부" 6회 발생')

tf = textbox(s, 6.75, 1.98, 6.1, 0.35)
para(tf, '② 원인 — 학습 데이터의 바위 박스 크기', 13.5, True, NAVY, first=True)
rows = [
    ['학습 데이터', '바위 개수', '면적 평균', '면적 최대'],
    ['v1 (82장)', '33개', '9.9%', '15.5%'],
    ['v3 (724장)', '294개', '12.1%', '40.4%'],
]
table(s, 6.75, 2.38, 6.1, rows, col_w=[1.8, 1.4, 1.4, 1.5], size=11,
      highlight_rows=(2,), row_h=0.36)
note(s, 6.75, 3.6, 6.1,
     '· v1 학습 데이터에 화면 15.5% 초과 바위 0장\n'
     '· 게임 시 손 거리에서는 20~40% → v1에게는 미경험 크기\n'
     '· 클래스별 개수는 34/33/33으로 균형 — 개수의 문제 아님')

tf = textbox(s, 0.5, 4.5, 12.3, 0.35)
para(tf, '③ 검증 — 평가 지표가 동일한 결론을 지시', 13.5, True, NAVY, first=True)
rows = [
    ['모델', '바위 Precision', '바위 Recall', '바위 mAP50', '해석'],
    ['v1', '0.918', '0.352', '0.514', '"바위"라고 판단하면 대부분 정확하나 미검출이 다수'],
    ['v3', '0.930', '0.775', '0.908', '미검출 비율 절반 이하로 감소'],
]
table(s, 0.5, 4.9, 12.3, rows, col_w=[1.0, 1.7, 1.6, 1.6, 6.4], size=11,
      highlight_rows=(2,), row_h=0.42)

note(s, 0.5, 6.2, 12.3,
     '④ 적용 결과 — 바위 Recall 0.352 → 0.775 (2.2배), 보드 판정 정확도 3/9 → 9/9\n'
     '   세 모양 중 주먹이 면적 최소(평균 9.9%, 가위 14.2%·보 18.5%)로 검출 난이도 최상, '
     '학습 데이터의 크기 범위도 최협소')

# ═══════════════════════════════════════════════════════════
# 4-4. 보드 실측 + 기대 효과
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '4', '결과 분석 및 기대 효과', '4-4. 보드 실측과 기대 효과')

conclusion_box(s, 0.5, 1.2, 12.3,
               '보드 실측 — 9가지 조합 전부 정답(9/9). 같은 자리·같은 조건에서 v1은 3/9', size=16)

picture(s, IMG_V1TERM, 0.5, 1.95, w=6.0)
caption(s, 0.5, 2.72, 6.0, '▲ 그림 5. v1 (82장 학습) — P1 2 / 무 6 / P2 1 → 3판만 정답')
picture(s, IMG_V3TERM, 6.8, 1.95, w=6.0)
caption(s, 6.8, 2.72, 6.0, '▲ 그림 6. v3 (724장 학습) — P1 3 / 무 3 / P2 3 → 9판 전부 정답')

tf = textbox(s, 0.5, 3.1, 12.3, 0.35)
para(tf, '목표 달성 여부', 15, True, NAVY, first=True)
rows = [
    ['항목', '목표', '결과', '달성'],
    ['5단계 과제 구현', '전부', '전부 구현', 'O'],
    ['보드 단독 실시간 동작', '10 FPS 이상', '68.1 FPS (INT8 · MJPG)', 'O'],
    ['승패 판정 정확도', '9가지 조합 전부', '9 / 9', 'O'],
    ['실사용 인식 성능', '—', 'mAP50 0.853 (v1 0.511)', 'O'],
]
table(s, 0.5, 3.55, 12.3, rows, col_w=[3.0, 2.6, 4.7, 1.0], size=11.5, row_h=0.38)

tf = textbox(s, 0.5, 5.6, 12.3, 1.7)
para(tf, '기대 효과', 15, True, NAVY, first=True)
para(tf, '· 정해진 손 모양 판별 장치로 이전 가능 — 무인 키오스크 제스처 입력, 재활 훈련 동작 확인, 비접촉 조작 패널',
     13, space_before=6)
para(tf, '· 보드 단독으로 검출·판정·표시 완결 → 네트워크 불필요, 영상이 기기 외부로 미전송',
     13)
para(tf, '· 데이터셋 도구(make_dataset.py, check_labels.py)는 클래스만 교체하면 타 검출 과제에 재사용 가능',
     13)

# ═══════════════════════════════════════════════════════════
# 4-5. 라벨 품질 수치화
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '4', '결과 분석 및 기대 효과',
             '4-5. 라벨 품질을 수치화하고, 그 영향을 실험으로 확인 — 가설 기각')

conclusion_box(s, 0.5, 1.2, 12.3,
               '라벨이 크게 다른 것은 사실(IoU 0.693)이나, 성능 차이는 없었다 — "이상하다"와 "성능을 떨어뜨린다"는 별개',
               size=15)

tf = textbox(s, 0.5, 1.95, 6.1, 0.35)
para(tf, '① 같은 사진 88장을 두 사람이 라벨링', 14, True, NAVY, first=True)

tf = textbox(s, 0.5, 2.33, 6.1, 0.9)
para(tf, '· 팀원 txt는 보지 않고 사진만 복사해 다시 라벨링', 11.5, first=True)
para(tf, '· 클래스별 균등 추출, 섞여 있던 교수님 원본 12장은 해시로 제외', 11.5)

rows = [
    ['항목', '값'],
    ['클래스가 다른 것', '0장'],
    ['IoU 평균 / IoU<0.5', '0.693 / 17장 (19%)'],
    ['팀원 박스 평균 면적', '+52% (보는 2배)'],
]
table(s, 0.5, 3.25, 6.1, rows, col_w=[2.6, 3.5], size=11.5,
      highlight_rows=(3,), row_h=0.33)

tf = textbox(s, 0.5, 4.75, 6.1, 0.35)
para(tf, '② A/B 실험 — 사진 동일, 라벨만 교체', 14, True, NAVY, first=True)

tf = textbox(s, 0.5, 5.13, 6.1, 0.6)
para(tf, '각 257장 · 80 epoch · 설정 동일. 차이는 88장의 라벨뿐', 11.5, first=True)

rows = [
    ['시험지 (박스 수)', 'A 팀원', 'B 다시', '차이'],
    ['교수님 22장 (32)', '0.849', '0.857', '+0.008'],
    ['hold40 48장 (86)', '0.841', '0.831', '−0.010'],
    ['오늘 233장 (328)', '0.815', '0.814', '−0.001'],
    ["〃 '보'만", '0.848', '0.852', '+0.004'],
]
table(s, 0.5, 5.68, 6.1, rows, col_w=[2.3, 1.3, 1.3, 1.2], size=10.5,
      highlight_rows=(3,), row_h=0.28)
note(s, 0.5, 6.98, 6.1, '값 = mAP50-95 (mAP50은 네 경우 모두 0.995로 동일)')

picture(s, IMG_LABELVS, 6.85, 1.95, w=5.9)
caption(s, 6.85, 6.45, 5.9,
        '▲ 그림 5. 같은 사진의 두 라벨 — 빨강: 팀원 / 초록: 다시 라벨링')

tf = textbox(s, 6.85, 6.62, 6.1, 0.65)
para(tf, '고찰 — 측정한 사실과 거기서 추론한 것은 다르다', 13, True, NAVY, first=True)
para(tf, '· 박스가 팔뚝·책상까지 감싼 것은 참, 성능을 깎는다는 것은 기각(3개 시험지)',
     11, True, RED)

# ═══════════════════════════════════════════════════════════
# 4-6. 신뢰도 문턱값
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '4', '결과 분석 및 기대 효과',
             '4-6. 재학습 없이 게임 판정률 3.1배 — 원인은 상수 하나')

conclusion_box(s, 0.5, 1.2, 12.3,
               'mAP50 0.859인 모델이 게임에서는 자세 4개 중 3개에서 판정 불가 — 학습이 아니라 후처리 문턱값 문제였다',
               size=15)

tf = textbox(s, 0.5, 1.95, 6.1, 0.35)
para(tf, '① 모순처럼 보인 두 수치', 14, True, NAVY, first=True)
rows = [
    ['측정', '값'],
    ['mAP50 (문턱값 없음)', '0.859'],
    ['conf 0.35에서 두 손 검출', '32%'],
]
table(s, 0.5, 2.35, 6.1, rows, col_w=[3.3, 2.8], size=12,
      highlight_rows=(2,), row_h=0.34)
note(s, 0.5, 3.5, 6.1,
     '· 박스는 나오는데 신뢰도가 낮아 잘렸다. mAP는 모든 신뢰도 구간을\n'
     '  훑어 채점하지만, 게임은 CONF_TH 한 값으로 자른다')

tf = textbox(s, 0.5, 4.15, 6.1, 0.35)
para(tf, '② 게임의 판정 절차를 그대로 재현해 측정', 14, True, NAVY, first=True)

tf = textbox(s, 0.5, 4.53, 6.1, 1.0)
para(tf, '박스를 신뢰도 순 2개로 추림 → x좌표로 좌/우 구분', 12, first=True)
para(tf, '→ 정답 쌍과 대조 (시험지 120장, 두 손)', 12)
para(tf, '※ "박스가 2개 나왔나"만 세면 얼굴 박스도 성공으로 계수됨', 11.5, True, RED)

rows = [
    ['문턱값', '판정 정확도', '판정을 내린 자세', '배경 오검출'],
    ['0.40 (원래)', '90.3%', '25.8%', '0개'],
    ['0.25', '79.2%', '44.2%', '0개'],
    ['0.15', '80.0%', '66.7%', '0개'],
    ['0.10 (적용)', '78.4%', '80.8%', '0개'],
    ['0.05', '72.4%', '96.7%', '0개'],
]
table(s, 6.85, 1.95, 6.1, rows, col_w=[1.6, 1.5, 1.6, 1.4], size=11.5,
      highlight_rows=(4,), row_h=0.34)
note(s, 6.85, 4.1, 6.1,
     '· 0.35~0.10 구간에서 정확도는 78~81%로 평평한데 판정 비율만 2.6배 상승\n'
     '· 0.05에서 정확도가 72.4%로 꺾이므로 0.10이 최적점')

tf = textbox(s, 6.85, 4.75, 6.1, 0.35)
para(tf, '③ 적용 결과 (RPS_Game.py 2줄 수정)', 14, True, NAVY, first=True)
rows = [
    ['', 'Before', 'After'],
    ['CONF_TH / USE_TOP2', '0.4 / False', '0.10 / True'],
    ['두 손 검출 (120장)', '31장', '97장'],
    ['판정을 내린 자세', '25.8%', '80.8%'],
    ['재학습', '—', '없음'],
]
table(s, 6.85, 5.15, 6.1, rows, col_w=[2.5, 1.8, 1.8], size=11.5,
      highlight_rows=(3,), row_h=0.34)

tf = textbox(s, 0.5, 5.75, 6.1, 1.5)
para(tf, '고찰', 14, True, NAVY, first=True)
para(tf, '· 모델 점수와 제품 성능은 다른 축 — 재학습 전에 후처리를 먼저 점검', 12, space_before=4)
para(tf, '· 중간 지표("박스 2개")를 최적화하면 0.05가 최적으로 보임', 12)
para(tf, '· 측정은 제품이 실제로 하는 일 그대로 해야 한다', 12, True, GREEN)

# ═══════════════════════════════════════════════════════════
# 5. 향후 연구 과제
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '5', '향후 연구 과제')

conclusion_box(s, 0.5, 1.25, 12.3,
               '남은 과제는 모두 원인이 수치로 특정되었거나, 규명에 필요한 실험이 설계·준비되어 있음',
               size=15)

rows = [
    ['과제', '현재 상태 (수치)', '원인', '해결 방향'],
    ['v9 성능 저하 원인',
     'v3 724장 0.853 → v9 1,318장 0.643\n(오늘 233장 시험지, mAP50)',
     '라벨 품질은 A/B 실험으로 기각\n원인 미특정',
     '추가분 434장의 성질별 분리 실험\n(손 개수 / 박스 크기 분포 / 촬영 환경)'],
    ['얼굴을 손으로 오검출',
     '학습 데이터의 background 사진 12장 / 1,318장 (0.9%)',
     '박스 밖을 "없음"으로 학습하나\n여백에 얼굴이 거의 없음',
     '손 없는 사진을 전체의 10% 수준까지 추가\n(문턱값 0.10 에서 배경 25장 오검출 0개는 확인)'],
    ['박스 위치 정밀도',
     '실사용 mAP50 0.853 대비 mAP50-95 0.448',
     '라벨링 담당자별 박스 기준 상이\n(위 항목과 같은 뿌리)',
     '라벨링 기준 문서화 후 재라벨링, 또는 상위 모델(YOLO11s)'],
    ['외부 데이터 교차 검증',
     '자체 시험지는 우리 라벨 기준에 맞춰져 있어 편향이 남음',
     '공개 데이터셋(Roboflow RPS 3,129장)\n미사용',
     '단, 그쪽 박스 기준이 우리와 같은지 먼저 검증해야\n낮은 점수가 성능인지 기준 차이인지 구분 가능'],
]
table(s, 0.5, 2.1, 12.3, rows, col_w=[2.0, 3.0, 3.3, 4.0], size=9.5, row_h=0.85)

note(s, 0.5, 5.75, 12.3,
     '· A/B 실험 실행 완료 — 라벨 품질 가설은 기각됨(시험지 3개에서 차이 0.001~0.010). 따라서 v9의 성능 저하 원인은 다시 미해결\n'
     '· 다음 후보: 추가된 434장이 전부 손 1개이고 주먹 박스 평균 5.2%로 매우 작다는 점 (분포 이동). 요인별 분리 실험 필요')

# ═══════════════════════════════════════════════════════════
# 6. 후기
# ═══════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
slide_header(s, '6', '프로젝트 수행 후기')

conclusion_box(s, 0.5, 1.25, 12.3, '가장 크게 배운 것 — "측정하지 않으면 모른다"', size=17)

rows = [
    ['배운 것', '무엇을 겪었나', '수치'],
    ['추측은 자주 틀린다',
     '후처리 40줄이 병목일 것으로 예상 → 실제 병목은 한 줄짜리 waitKey',
     '후처리 0.36 ms / waitKey 11.98 ms'],
    ['높은 점수를 의심해야 한다',
     'mAP 0.995를 완성으로 판단했으나 어두운 옷 앞에서 실패. 시험지와 학습 조건이 동일했음',
     '같은 모델이 실사용 조건에서 0.511'],
    ['데이터가 모델을 정한다',
     '모델은 한 번도 바꾸지 않고 데이터만 늘렸음. 단 양이 아니라 "시험 볼 조건이 학습에 들어 있는가"가 성능을 정함',
     '104 → 724장에 실사용 mAP50 0.511 → 0.853'],
    ['무음 실패가 가장 위험하다',
     '학습이 2 epoch에서 끊긴 모델을 80 epoch 모델과 비교하고 "데이터가 나쁘다"고 결론지었음. 어떤 경고도 없었음',
     '결론 철회 후 epoch 검사 코드 추가'],
]
table(s, 0.5, 2.1, 12.3, rows, col_w=[2.4, 6.4, 3.5], size=11, row_h=0.75)

tf = textbox(s, 0.5, 5.6, 12.3, 1.6)
para(tf, '마무리', 15, True, NAVY, first=True)
para(tf, '· 대학원 산학 프로젝트에서 NPU SDK가 제공하던 성능 타임라인을 직접 구현 → 당시 보던 그림의 의미를 이해',
     13, space_before=6)
para(tf, '· 비교 가능한 실험 설계(학습 조건 고정, 시험지 보존)가 결론의 신뢰도를 결정함을 확인',
     13)
para(tf, '· 남은 과정에서도 "먼저 측정하고, 비교 가능하게 설계한다"를 이어갈 계획', 13)

# ── 넘침 검사 : 슬라이드(7.5in) 밖으로 나간 도형이 있으면 알려 준다 ──
LIMIT = Inches(7.5)
bad = []
for i, sl in enumerate(prs.slides, 1):
    for sh in sl.shapes:
        if sh.top is None or sh.height is None:
            continue
        bottom = sh.top + sh.height
        if bottom > LIMIT:
            over = (bottom - LIMIT) / 914400.0
            bad.append((i, sh.shape_type, round(over, 2)))
if bad:
    print()
    print('!! 슬라이드 밖으로 넘친 도형 %d개' % len(bad))
    for i, t, o in bad:
        print('   슬라이드 %d : %s  %.2f in 초과' % (i, t, o))
else:
    print('넘침 검사 : 이상 없음')

prs.save(OUT)
print('생성 완료 :', os.path.abspath(OUT))
print('슬라이드 %d장' % len(prs.slides._sldIdLst))
