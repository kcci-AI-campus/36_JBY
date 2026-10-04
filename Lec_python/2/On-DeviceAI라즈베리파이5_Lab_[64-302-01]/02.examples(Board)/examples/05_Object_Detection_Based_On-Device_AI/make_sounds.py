# -*- coding: utf-8 -*-
"""
승/패/무 효과음 wav 세 개를 만든다. (5단계 '승패 효과음'용)

효과음 파일을 어디서 구하지 않고 직접 만든다. 소리는 결국 공기의 떨림이고,
wav는 그 떨림의 높낮이를 1초에 44,100번 적어 둔 숫자 목록일 뿐이다.
사인파(sin)를 그 숫자로 적으면 '삐-' 하는 순음이 된다.

  win.wav   도 → 미 → 솔  (올라가는 소리)
  lose.wav  솔 → 미 → 도  (내려가는 소리)
  draw.wav  미 한 번       (짧게 한 음)

실행:  python3 make_sounds.py
확인:  aplay win.wav
"""
import wave
import struct
import math

RATE = 44100          # 1초에 숫자 44,100개 (CD 음질)
AMP = 12000           # 소리 크기 (16비트라 최대 32767)

# 음이름과 진동수(Hz). 1초에 몇 번 떨리는지가 음 높이다.
NOTE = {'도': 523.25, '미': 659.25, '솔': 783.99}


def tone(freq, sec):
    """freq Hz 사인파를 sec초만큼 만든다. 양 끝은 부드럽게 줄여 '탁' 소리를 없앤다."""
    n = int(RATE * sec)
    out = []
    fade = int(RATE * 0.01)        # 10ms 페이드
    for i in range(n):
        v = math.sin(2 * math.pi * freq * i / RATE)
        if i < fade:
            v *= i / fade                  # 시작할 때 서서히 키우고
        elif i > n - fade:
            v *= (n - i) / fade            # 끝날 때 서서히 줄인다
        out.append(int(AMP * v))
    return out


def save(path, notes, sec=0.12):
    """음 이름 목록을 이어 붙여 wav 파일로 저장한다."""
    data = []
    for nm in notes:
        data += tone(NOTE[nm], sec)
    with wave.open(path, 'w') as w:
        w.setnchannels(1)          # 모노
        w.setsampwidth(2)          # 16비트 = 2바이트
        w.setframerate(RATE)
        w.writeframes(b''.join(struct.pack('<h', v) for v in data))
    print('  %-10s %s  (%.2f초)' % (path, ' '.join(notes), len(data) / RATE))


print('효과음을 만든다')
save('win.wav',  ['도', '미', '솔'])
save('lose.wav', ['솔', '미', '도'])
save('draw.wav', ['미'], sec=0.2)
print()
print('들어 보기 :  aplay win.wav')
print('소리가 안 나면 출력 장치를 확인한다 :  aplay -l')
print('RPS_Game.py 의 SOUND 를 True 로 두면 판정할 때 자동으로 난다.')
