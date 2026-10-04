# ============================================================
#  v2e120 을 INT8 로 — 이번이 진짜 본선이다
#
#  왜 다시 하는가
#    어제 INT8 을 만들 때 원본으로 v2f_best.pt (merged_dataset_final, 80에폭) 를 썼다.
#    그런데 detect_rate.py 로 같은 60프레임을 15개 모델에 먹여 보니
#
#        v2e120_fp32_640   knife 100 %  (신뢰도 0.68)
#        v2f_fp32_640      knife  25 %  (신뢰도 0.28)   <- 원본부터 약하다
#        v2f_int8_640      knife   0 %                  <- 남은 것마저 사라졌다
#
#    **원본이 약해서 INT8 이 0% 가 된 것**이지 양자화 자체의 문제가 아니다.
#    속도는 v2f_int8_640 으로 이미 증명됐다 (21 FPS, CPU 72%).
#    이제 그 속도를 '칼을 100% 찾는 가중치' 위에서 다시 얻어야 한다.
#
#  무엇을 만드는가
#    v2e120_best.pt (에폭 120) 를 416 / 512 / 640 세 크기로 INT8 변환한다.
#    640 이 목표이고, 속도가 모자라면 512·416 으로 내려갈 수 있게 같이 만든다.
#
#  Colab 에 새 셀을 만들고 아래를 통째로 붙여넣어 실행하세요.
# ============================================================

# ── 1단계 : 환경 ──────────────────────────────────────────
!pip -q install "ultralytics==8.4.98"
!pip -q install ai-edge-litert

import os, glob, shutil, zipfile
from google.colab import files

OUT = '/content/out_v2e120'
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
#   C:\Users\kccistc\Developer\침입감지_프로젝트\v2e120_best.pt   (5.19 MB)
#   (ptfile.zip 안의 120/best.pt 를 꺼내 둔 것입니다. 이미 만들어 뒀습니다.)
print('가중치(v2e120_best.pt)를 고르세요')
up = files.upload()
PT = list(up.keys())[0]
print('올린 가중치 :', PT, '%.2f MB' % (os.path.getsize(PT) / 1048576))


# ── 3단계 : 교정용 데이터 올리기 ──────────────────────────
#   INT8 양자화는 '이런 그림이 들어온다' 를 보여 줄 사진이 필요하다(교정, calibration).
#   v2e120 이 어떤 데이터로 학습됐는지는 팀원에게 확인해야 하지만,
#   교정은 '비슷한 장면' 이면 되므로 merged_dataset_final 로 충분하다.
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

n_val = len(glob.glob(os.path.join(ROOT, val_dir, '*')))
print('교정에 쓸 사진 :', n_val, '장  (도구 권장은 300장 이상)')


# ── 4단계 : INT8 변환 ─────────────────────────────────────
SIZES = [416, 512, 640]

for SZ in SIZES:
    print()
    print('=' * 60)
    print(' INT8 내보내기   v2e120   imgsz =', SZ)
    print('=' * 60)
    clear_tflite()
    !yolo export model={PT} format=litert imgsz={SZ} data={DATA} quantize=8
    take_newest('v2e120_int8_%d.tflite' % SZ)


# ── 5단계 : 비교용 FP32 도 하나 (512 는 아직 없다) ────────
print()
print('=' * 60)
print(' FP32 내보내기   v2e120   imgsz = 512   (INT8 512 와 짝 비교용)')
print('=' * 60)
clear_tflite()
!yolo export model={PT} format=litert imgsz=512
take_newest('v2e120_fp32_512.tflite')


print()
print('만든 것')
for f in sorted(glob.glob(OUT + '/*.tflite')):
    print('  %8.2f MB  %s' % (os.path.getsize(f) / 1048576, os.path.basename(f)))


# ── 6단계 : 내려받기 ──────────────────────────────────────
!cd /content && rm -f v2e120_all.zip && zip -qj v2e120_all.zip out_v2e120/*.tflite
files.download('/content/v2e120_all.zip')

print("""
============================================================
 보드에서 할 것  (~/work/baby/ 에 넣고)
============================================================
 1) 검출 — 같은 장면을 한 번 찍어 전부에 먹인다
      python3 detect_rate.py --wait 10 --save

    보고 싶은 것은 이 네 줄이다 :
      v2e120_fp32_640   knife 100 %   <- 기준선 (이미 확인됨)
      v2e120_int8_640   knife  ?? %   <- 여기가 100% 에 가까우면 끝
      v2e120_int8_512   knife  ?? %
      v2e120_int8_416   knife  ?? %

    640 이 떨어지면 512 -> 416 순으로 본다.
    ** 이번에는 인형(baby)도 반드시 화면에 넣을 것. **
       지난 측정은 인형이 안 들어와 baby 가 전 모델 0% 로 나왔다.

 2) 속도 — 실사용 기준으로
      python3 detect_baby.py v2e120_int8_640.tflite --web --seconds 30

    v2f_int8_640 이 21.1 FPS / CPU 72% 였으므로 같은 값이 나와야 정상이다
    (구조가 같은 yolo11n 이라 가중치 값은 속도에 영향이 없다).

 3) 오검출 — 칼과 인형을 모두 치우고 한 번 더
      python3 detect_rate.py --wait 10 --save

    여기서 나오는 knife / outlet 검출률이 전부 오검출이다.
    (책을 knife 46% 로 잡은 전례가 있다. 아직 한 번도 측정 안 했다.)
============================================================
""")
