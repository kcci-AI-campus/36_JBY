# Bench 5 models x (FP32/INT8): camera 30 fps real-use loop (capture + preprocess + invoke), PMIC power, CPU.
# usage: python3 bench_power.py <seconds> <tflite...>   (run inside ~/work/env)
import sys, os, time, re, glob, threading, subprocess
import numpy as np, cv2
from ai_edge_litert.interpreter import Interpreter

SEC = float(sys.argv[1]); MODELS = sys.argv[2:]

def pmic_w():
    out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True).stdout
    a, v = {}, {}
    for m in re.finditer(r"(\S+?)_A\s+current\(\d+\)=([\d.]+)A", out): a[m.group(1)] = float(m.group(2))
    for m in re.finditer(r"(\S+?)_V\s+volt\(\d+\)=([\d.]+)V", out): v[m.group(1)] = float(m.group(2))
    return sum(a[k] * v[k] for k in a if k in v)

def cpu_ticks():
    f = open("/proc/stat").readline().split()[1:]
    vals = list(map(int, f)); return sum(vals), vals[3] + vals[4]

class Sampler(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.ws = []; self.stop = False
    def run(self):
        while not self.stop:
            try: self.ws.append(pmic_w())
            except Exception: pass
            time.sleep(1.0)

def temp():
    try: return float(re.search(r"([\d.]+)", subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout).group(1))
    except Exception: return -1

def measure_idle(sec=8):
    s = Sampler(); s.start(); time.sleep(sec); s.stop = True; s.join(); return float(np.mean(s.ws))

def run_model(path, cap):
    it = Interpreter(model_path=path, num_threads=4); it.allocate_tensors()
    inp = it.get_input_details()[0]; shp = inp["shape"]; dt = inp["dtype"]
    nchw = (len(shp) == 4 and shp[1] == 3)
    S = int(shp[2]) if nchw else int(shp[1])
    q = inp.get("quantization", (0.0, 0)); intin = dt in (np.int8, np.uint8)
    # warm-up
    dummy = np.zeros(shp, dtype=dt); it.set_tensor(inp["index"], dummy)
    for _ in range(3): it.invoke()
    t_end = time.time() + SEC; n = 0; inv = []; c0 = cpu_ticks(); s = Sampler(); s.start(); t0 = time.time()
    while time.time() < t_end:
        ok, f = cap.read()
        if not ok: continue
        H, W = f.shape[:2]; r = min(S / W, S / H); nw, nh = int(W * r), int(H * r)
        x = np.full((S, S, 3), 114, np.uint8); px, py = (S - nw) // 2, (S - nh) // 2
        x[py:py + nh, px:px + nw] = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (nw, nh))
        if intin:
            sc, zp = q; xx = np.clip(np.round((x.astype(np.float32) / 255.0) / (sc if sc else 1.0) + zp), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
        else:
            xx = x.astype(np.float32) / 255.0
        xx = xx[None]
        if nchw: xx = np.ascontiguousarray(np.transpose(xx, (0, 3, 1, 2)))
        it.set_tensor(inp["index"], xx); t1 = time.time(); it.invoke(); inv.append((time.time() - t1) * 1000); n += 1
    el = time.time() - t0; s.stop = True; s.join(); c1 = cpu_ticks()
    cpu = 100.0 * (1 - (c1[1] - c0[1]) / max(1, c1[0] - c0[0]))
    return dict(model=os.path.basename(path), mb=os.path.getsize(path) / 1e6, input=S, dtype=("int8" if intin else "fp32"),
                invoke_ms=float(np.mean(inv)), fps=n / el, w=float(np.mean(s.ws)), cpu=cpu, temp=temp(), n=n)

cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
for _ in range(10): cap.read()
idle = measure_idle(); print("idle_w(camera open, no inference) %.2f" % idle, flush=True)
rows = []
for p in MODELS:
    try:
        r = run_model(p, cap); r["idle"] = idle; r["dyn_w"] = r["w"] - idle; r["mj_per_inf"] = r["dyn_w"] / r["fps"] * 1000 if r["fps"] else 0
        rows.append(r)
        print("%-30s %5.1fMB in%3d %s invoke %6.1fms fps %5.1f W %.2f (dyn %.2f) mJ/inf %6.1f cpu %4.1f%% temp %.1f n=%d" % (
            r["model"], r["mb"], r["input"], r["dtype"], r["invoke_ms"], r["fps"], r["w"], r["dyn_w"], r["mj_per_inf"], r["cpu"], r["temp"], r["n"]), flush=True)
    except Exception as e:
        print("FAIL", p, e, flush=True)
    time.sleep(5)
cap.release()
import csv
with open("bench_power_5model.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print("saved bench_power_5model.csv")
