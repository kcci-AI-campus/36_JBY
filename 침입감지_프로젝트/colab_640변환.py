# ============================================================
#  입력 크기를 바꿔 다시 내보내기 (재학습 없음)
#
#  왜 하는가
#    2m 거리의 커터칼이 320 입력에서는 약 21픽셀밖에 안 된다.
#    그 크기에서는 어떤 모델도 찾기 어렵다.
#    입력을 640 으로 올리면 42픽셀이 되어 유효 거리가 2배가 된다.
#    ** 가중치는 이미 640 으로 학습돼 있다. 내보내기만 다시 하면 된다. **
#
#  무엇을 비교하는가
#    같은 가중치를 320 과 640 으로 각각 내보내, 보드에서 같은 장면으로 잰다.
#    가중치가 같으므로 차이는 오직 입력 크기뿐이다 = 통제된 비교.
#
#  이름 겹침 주의
#    yolo export 는 매번 best.tflite 라는 같은 이름으로 만든다.
#    그래서 내보내기 직전에 기존 tflite 를 지우고,
#    새로 생긴 것만 찾아 '에폭_크기' 가 드러나는 이름으로 바꾼다.
#
#  Colab 에 새 셀을 만들고 아래를 통째로 붙여넣어 실행하세요.
# ============================================================

# ── 1단계 : 환경 ──────────────────────────────────────────
!pip -q install "ultralytics==8.4.98"
!pip -q install ai-edge-litert

import os, glob, shutil, zipfile
from google.colab import files

OUT = '/content/out_export'
os.makedirs(OUT, exist_ok=True)


def clear_tflite():
    """내보내기 전에 남아 있는 tflite 를 모두 지운다.
    그래야 '방금 만들어진 것' 을 확실히 집을 수 있다."""
    for f in glob.glob('/content/**/*.tflite', recursive=True):
        if OUT not in f:
            os.remove(f)


def take_newest(dst_name):
    """방금 만들어진 tflite 를 찾아 정해진 이름으로 옮긴다."""
    found = [f for f in glob.glob('/content/**/*.tflite', recursive=True)
             if OUT not in f]
    if not found:
        print('  !! tflite 가 만들어지지 않았습니다. 위 로그의 오류를 보세요.')
        return None
    src = max(found, key=os.path.getmtime)
    dst = os.path.join(OUT, dst_name)
    shutil.move(src, dst)
    print('  ->', dst_name, '%.2f MB' % (os.path.getsize(dst) / 1048576))
    return dst


# ── 2단계 : ptfile.zip 올리기 ─────────────────────────────
#   팀원이 준 파일입니다 (에폭별 best.pt 6개, 약 29 MB)
#       C:\Users\kccistc\Downloads\ptfile.zip
up = files.upload()
ZIP = list(up.keys())[0]
with zipfile.ZipFile(ZIP) as f:
    f.extractall('/content/pt')

PTS = {}
for p in sorted(glob.glob('/content/pt/**/best.pt', recursive=True)):
    ep = os.path.basename(os.path.dirname(p))      # 폴더 이름이 에폭
    PTS[ep] = p
print('찾은 가중치 :')
for ep, p in sorted(PTS.items(), key=lambda kv: int(kv[0])):
    print('  %4s 에폭   %.2f MB   %s' % (ep, os.path.getsize(p) / 1048576, p))


# ── 3단계 : 무엇을 내보낼지 정한다 ────────────────────────
#   먼저 한 가지 에폭만 320 / 640 으로 내보내 '해상도 효과' 를 확인한다.
#   에폭별 비교까지 하려면 EPOCHS 에 여러 개를 넣으면 된다(시간이 더 걸린다).
EPOCHS = ['120']            # 예: ['60', '100', '120']
SIZES = [320, 640]          # 같은 가중치를 두 크기로 -> 통제된 비교

print()
print('내보낼 것 : 에폭 %s  x  크기 %s  = %d개'
      % (EPOCHS, SIZES, len(EPOCHS) * len(SIZES)))


# ── 4단계 : FP32 로 내보내기 (교정 데이터 불필요) ─────────
#   "거리가 정말 늘어나는가" 만 먼저 확인하는 것이 목적이다.
#   FP32 는 640 에서 보드가 느리지만(약 10 FPS) 검출률 측정에는 지장이 없다.
#   속도는 6단계에서 INT8 로 해결한다.
for ep in EPOCHS:
    if ep not in PTS:
        print('!! 에폭', ep, '가 없습니다. 있는 것 :', sorted(PTS))
        continue
    PT = PTS[ep]
    for SZ in SIZES:
        print()
        print('=' * 60)
        print(' FP32 내보내기   에폭 %s   imgsz %d' % (ep, SZ))
        print('=' * 60)
        clear_tflite()
        !yolo export model={PT} format=litert imgsz={SZ}
        take_newest('v2e%s_fp32_%d.tflite' % (ep, SZ))

print()
print('만든 것')
for f in sorted(glob.glob(OUT + '/*.tflite')):
    print('  %8.2f MB  %s' % (os.path.getsize(f) / 1048576, os.path.basename(f)))


# ── 5단계 : 내려받아 보드에서 '거리' 확인 ─────────────────
!cd /content && rm -f export_fp32.zip && zip -qj export_fp32.zip out_export/*.tflite
files.download('/content/export_fp32.zip')

print("""
============================================================
 보드에서 할 것
============================================================
   받은 tflite 를 ~/work/baby/ 에 넣습니다.
   (v2e120_fp32_320 / v2e120_fp32_640 — 기존 파일과 이름이 안 겹칩니다)

   1) 칼과 인형을 **1.5~2m 거리**에 두고. 바닥에 테이프로 위치 표시!
        python3 detect_rate.py --wait 10 --save

      같은 60장을 320 과 640 에 똑같이 먹입니다.
      가중치가 같으므로 차이는 오직 입력 크기뿐입니다.

        v2e120_fp32_320   knife  5%   baby 50%
        v2e120_fp32_640   knife 70%   baby 85%   <- 이러면 해상도가 답이다

   2) 속도도 봅니다 (FP32 640 은 느릴 것입니다. 6단계에서 해결)
        python3 bench_models.py v2e120_fp32_640.tflite v2e120_fp32_320.tflite
============================================================
""")


# ── 6단계 : 거리가 늘어난 게 확인되면 INT8 로 (교정 데이터 필요) ──
#   INT8 양자화는 '이런 그림이 들어온다' 를 보여 줄 사진이 필요하다.
#   merged_v3.zip 을 올려야 한다. 아래 주석을 풀고 실행하세요.
#
# up2 = files.upload()          # merged_v3.zip 선택 (36 MB)
# for z in up2:
#     with zipfile.ZipFile(z) as f:
#         f.extractall('/content/data')
#
# ROOT = None
# for r, d, _ in os.walk('/content/data'):
#     if 'images' in d and 'labels' in d:
#         ROOT = r; break
# assert ROOT, '데이터 폴더를 못 찾았습니다.'
#
# CLASSES = ['baby', 'adult', 'knife', 'outlet']
# val_dir = 'images/val' if os.path.isdir(os.path.join(ROOT, 'images', 'val')) else 'images/test'
# lines = ['path: ' + ROOT, 'train: images/train', 'val: ' + val_dir, '',
#          'nc: ' + str(len(CLASSES)), 'names:']
# for i, c in enumerate(CLASSES):
#     lines.append('  ' + str(i) + ': ' + c)
# DATA = os.path.join(ROOT, 'data.yaml')
# open(DATA, 'w').write(chr(10).join(lines) + chr(10))
# print(open(DATA).read())
#
# for ep in EPOCHS:
#     PT = PTS[ep]
#     for SZ in SIZES:
#         print()
#         print('=' * 60)
#         print(' INT8 내보내기   에폭 %s   imgsz %d' % (ep, SZ))
#         print('=' * 60)
#         clear_tflite()
#         !yolo export model={PT} format=litert imgsz={SZ} data={DATA} quantize=8
#         take_newest('v2e%s_int8_%d.tflite' % (ep, SZ))
#
# print()
# print('최종 목록')
# for f in sorted(glob.glob(OUT + '/*.tflite')):
#     print('  %8.2f MB  %s' % (os.path.getsize(f) / 1048576, os.path.basename(f)))
#
# !cd /content && rm -f export_all.zip && zip -qj export_all.zip out_export/*.tflite
# files.download('/content/export_all.zip')
