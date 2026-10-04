# ============================================================
#  INT8 변환 — 640 을 실시간으로 돌리기 위한 마지막 단계
#
#  왜 하는가
#    640 입력이 검출 거리를 결정적으로 늘린다는 것은 확인됐다
#    (네 거리 모두 knife 100%, 320 은 전부 0%).
#    그런데 FP32 @640 은 보드에서 113 ms = 8 FPS 라 실시간이 불가능하다.
#    INT8 로 바꾸면 약 35 ms = 25 FPS 가 될 것으로 예상된다.
#
#    즉 이 프로젝트에서 INT8 은 '전력을 아끼려고' 쓰는 것이 아니라
#    **640 을 쓸 수 있게 해 주는 유일한 수단**이다.
#
#  무엇을 만드는가
#    같은 가중치(v2f_best.pt, merged_dataset_final 로 80에폭 640 학습)를
#    416 / 512 / 640 세 크기로 INT8 변환한다.
#    640 이 속도가 모자라면 512, 416 순으로 내려갈 수 있게 미리 만들어 둔다.
#
#  Colab 에 새 셀을 만들고 아래를 통째로 붙여넣어 실행하세요.
# ============================================================

# ── 1단계 : 환경 ──────────────────────────────────────────
!pip -q install "ultralytics==8.4.98"
!pip -q install ai-edge-litert

import os, glob, shutil, zipfile
from google.colab import files

OUT = '/content/out_int8'
os.makedirs(OUT, exist_ok=True)


def clear_tflite():
    """내보내기 전에 남은 tflite 를 지운다. 방금 만든 것을 확실히 집기 위해."""
    for f in glob.glob('/content/**/*.tflite', recursive=True):
        if OUT not in f:
            os.remove(f)


def take_newest(dst_name):
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


# ── 2단계 : 가중치 올리기 ─────────────────────────────────
#   C:\Users\kccistc\Developer\침입감지_프로젝트\v2f_best.pt   (5.2 MB)
print('가중치(v2f_best.pt)를 고르세요')
up = files.upload()
PT = list(up.keys())[0]
print('올린 가중치 :', PT, '%.2f MB' % (os.path.getsize(PT) / 1048576))


# ── 3단계 : 교정용 데이터 올리기 ──────────────────────────
#   INT8 양자화는 '이런 그림이 들어온다' 를 보여 줄 사진이 필요하다(교정, calibration).
#   학습에 쓴 것과 같은 데이터를 쓰는 것이 맞다.
#   C:\Users\kccistc\Downloads\merged_dataset_final.zip   (43.7 MB)
print()
print('교정용 데이터(merged_dataset_final.zip)를 고르세요')
up2 = files.upload()
for z in up2:
    with zipfile.ZipFile(z) as f:
        f.extractall('/content/data')

ROOT = None
for r, d, _ in os.walk('/content/data'):
    if 'images' in d and 'labels' in d:
        ROOT = r
        break
assert ROOT, '데이터 폴더를 못 찾았습니다.'
print('데이터 :', ROOT)

# data.yaml 을 Colab 경로로 다시 쓴다.
#   받은 zip 의 data.yaml 은 path 가 윈도우 경로(C:/...)로 남아 있어 그대로는 못 쓴다.
CLASSES = ['baby', 'adult', 'knife', 'outlet']
val_dir = 'images/val' if os.path.isdir(os.path.join(ROOT, 'images', 'val')) else 'images/test'
lines = ['path: ' + ROOT, 'train: images/train', 'val: ' + val_dir, '',
         'nc: ' + str(len(CLASSES)), 'names:']
for i, c in enumerate(CLASSES):
    lines.append('  ' + str(i) + ': ' + c)
DATA = os.path.join(ROOT, 'data.yaml')
open(DATA, 'w').write(chr(10).join(lines) + chr(10))
print()
print(open(DATA).read())


# ── 4단계 : INT8 변환 ─────────────────────────────────────
#   640 이 목표이고, 속도가 모자랄 때를 대비해 512·416 도 만들어 둔다.
SIZES = [416, 512, 640]

for SZ in SIZES:
    print()
    print('=' * 60)
    print(' INT8 내보내기   imgsz =', SZ)
    print('=' * 60)
    clear_tflite()
    !yolo export model={PT} format=litert imgsz={SZ} data={DATA} quantize=8
    take_newest('v2f_int8_%d.tflite' % SZ)

print()
print('만든 것')
for f in sorted(glob.glob(OUT + '/*.tflite')):
    print('  %8.2f MB  %s' % (os.path.getsize(f) / 1048576, os.path.basename(f)))


# ── 5단계 : 내려받기 ──────────────────────────────────────
!cd /content && rm -f int8_all.zip && zip -qj int8_all.zip out_int8/*.tflite
files.download('/content/int8_all.zip')

print("""
============================================================
 보드에서 할 것
============================================================
   v2f_int8_416 / 512 / 640 을 ~/work/baby/ 에 넣습니다.

   1) 속도 — 어느 크기까지 실시간이 되는가
        python3 bench_models.py v2f_int8_640.tflite v2f_int8_512.tflite v2f_int8_416.tflite

      한 프레임 33 ms 안에 들어와야 30 FPS 다.
      (INT8 @640 은 약 35 ms = 25 FPS 로 예상)

   2) 검출 — INT8 이 640 에서도 FP32 만큼 찾는가
      칼과 인형을 아까와 같은 자리에 두고
        python3 detect_rate.py --wait 10 --save

      v2f_fp32_640 과 v2f_int8_640 을 나란히 비교합니다.
      비슷하면 INT8 채택, 크게 떨어지면 512 로 내려갑니다.

   3) 오검출 — 칼과 인형을 모두 치우고 한 번 더
        python3 detect_rate.py --wait 10 --save

      여기서 나오는 knife / outlet 검출률이 전부 오검출입니다.
      문턱값을 얼마로 올려야 사라지는지 세 값(0.25/0.40/0.50)으로 한 번에 봅니다.
============================================================
""")
