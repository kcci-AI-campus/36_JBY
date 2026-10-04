#!/bin/bash
# H. 유휴(아무것도 안 돌릴 때) 전력 바닥 낮추기 — 설정 하나씩 바꿔 가며 PMIC 25초 평균
#   되돌리기까지 포함. 실행: bash h_idle.sh
set -u
POL=/sys/devices/system/cpu/cpufreq/policy0
MAXF=$(cat $POL/cpuinfo_max_freq)
measure() {  # measure <이름>
  python3 - "$1" <<'PY'
import subprocess, re, time, sys
name=sys.argv[1]; vals=[]; t0=time.time()
while time.time()-t0 < 25:
    out=subprocess.run(["vcgencmd","pmic_read_adc"],capture_output=True,text=True).stdout
    a={};v={}
    for m in re.finditer(r"(\S+?)_([AV])\s+\w+\(\d+\)=([\d.]+)[AV]",out):
        (a if m.group(2)=="A" else v)[m.group(1)]=float(m.group(3))
    w=sum(a[n]*v[n] for n in a if n in v)
    if w>0: vals.append(w)
    time.sleep(1)
vals.sort(); n=len(vals)
core=subprocess.run(["bash","-c","cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq"],capture_output=True,text=True).stdout.strip()
print("  %-34s 평균 %.3f W  중앙 %.3f  최소 %.3f  최대 %.3f  (표본 %d, 현재클럭 %s kHz)" % (name, sum(vals)/n, vals[n//2], vals[0], vals[-1], n, core))
PY
}
echo "===== H. 유휴 전력 바닥 (아무 프로그램도 안 돌림) ====="
echo "wifi power_save: $(iw dev wlan0 get power_save 2>/dev/null)"; echo "bluetooth: $(rfkill list bluetooth 2>/dev/null | grep -i soft)"; echo "desktop: $(systemctl is-active lightdm 2>/dev/null)"
measure "H0 기본 (거버너 ondemand, 2400 MHz 상한)"
echo $((1500000)) | sudo -n tee $POL/scaling_max_freq >/dev/null
measure "H1 클럭 상한 1500 MHz"
sudo -n iw dev wlan0 set power_save on 2>/dev/null
measure "H2 + Wi-Fi 절전 켬"
sudo -n rfkill block bluetooth 2>/dev/null
measure "H3 + 블루투스 끔"
if systemctl is-active lightdm >/dev/null 2>&1; then sudo -n systemctl stop lightdm; sleep 3; measure "H4 + 데스크톱 세션(lightdm) 정지"; DESK=1; else echo "  (lightdm 이 안 돌고 있어 H4 생략)"; DESK=0; fi
echo powersave | sudo -n tee $POL/scaling_governor >/dev/null 2>&1 && measure "H5 + 거버너 powersave (최저 클럭 고정)"
# ── 되돌리기 ──
echo ondemand | sudo -n tee $POL/scaling_governor >/dev/null 2>&1
echo $MAXF | sudo -n tee $POL/scaling_max_freq >/dev/null
sudo -n rfkill unblock bluetooth 2>/dev/null
[ "${DESK:-0}" = "1" ] && sudo -n systemctl start lightdm
sleep 5
measure "H6 되돌린 뒤 (기본 상태 재확인)"
echo "사용 가능 주파수: $(awk '{print $1}' /sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state 2>/dev/null | tr '\n' ' ')"
echo "===== H 끝 ====="
