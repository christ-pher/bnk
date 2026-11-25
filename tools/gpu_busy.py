"""How busy each GPU is while a bnk server answers a request (any engine): NVML's utilization samples (the share
of time a kernel was running, ~6 samples a second) and memory-controller samples, split into the prompt read and
the decode.

    .venv/bin/python tools/gpu_busy.py [--url http://localhost:8080] [--prompt-chars 20000] [--max-tokens 400]
    .venv/bin/python tools/gpu_busy.py --watch 30        # no request: whatever the GPUs run for 30 s

A request's phases come from the stream: the prompt read lasts until the first token (reasoning or content), the
decode from there to the end. Thinking is off unless --think, so the decode is the answer's."""
import argparse
import ctypes
import json
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _Val(ctypes.Union):
    _fields_ = [("d", ctypes.c_double), ("ui", ctypes.c_uint), ("ul", ctypes.c_ulong), ("ull", ctypes.c_ulonglong)]


class _Sample(ctypes.Structure):
    _fields_ = [("ts", ctypes.c_ulonglong), ("v", _Val)]


class Nvml:
    GPU, MEM = 1, 2   # NVML_GPU_UTILIZATION_SAMPLES, NVML_MEMORY_UTILIZATION_SAMPLES

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2():
            raise SystemExit("nvmlInit failed")
        n = ctypes.c_uint()
        self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
        self.handles = []
        for i in range(n.value):
            h = ctypes.c_void_p()
            self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
            self.handles.append(h)

    def samples(self, i, kind, since):
        """[(timestamp us, percent)] newer than `since`."""
        vt, cnt = ctypes.c_int(), ctypes.c_uint(0)
        h = self.handles[i]
        if self.lib.nvmlDeviceGetSamples(h, kind, ctypes.c_ulonglong(since), ctypes.byref(vt), ctypes.byref(cnt), None):
            return []
        buf = (_Sample * max(cnt.value, 1))()
        if self.lib.nvmlDeviceGetSamples(h, kind, ctypes.c_ulonglong(since), ctypes.byref(vt), ctypes.byref(cnt), buf):
            return []
        return [(buf[k].ts, buf[k].v.ui) for k in range(cnt.value) if buf[k].ts > since]


class Sampler(threading.Thread):
    """Collects every GPU's samples (NVML keeps a short ring of them, so poll often)."""

    def __init__(self, nv):
        super().__init__(daemon=True)
        self.nv, self.stop = nv, threading.Event()
        self.data = {(i, k): {} for i in range(len(nv.handles)) for k in (Nvml.GPU, Nvml.MEM)}
        self.since = {key: 0 for key in self.data}

    def run(self):
        while not self.stop.is_set():
            for key in self.data:
                for ts, v in self.nv.samples(key[0], key[1], self.since[key]):
                    self.data[key][ts] = v
                    self.since[key] = max(self.since[key], ts)
            time.sleep(0.05)

    def mean(self, i, kind, t0, t1):
        """Average percent over samples in [t0, t1) (seconds, time.time())."""
        vals = [v for ts, v in self.data[(i, kind)].items() if t0 * 1e6 <= ts < t1 * 1e6]
        return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def request(url, chars, max_tokens, think, seed=None, note=None):
    files = sorted((ROOT / "src").glob("**/*.cpp"))
    ctx, n = [], 0
    for f in files:
        t = f.read_text(errors="replace")
        ctx.append(f"=== {f.name} ===\n{t}\n")
        n += len(ctx[-1])
        if n >= chars:
            break
    body = {"messages": [{"role": "system", "content": "You review C++ code.\n\n" + "".join(ctx)[:chars]},
                         {"role": "user", "content": f"Describe what these files do. Note: {time.time() if note is None else note}"}],
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": think}}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    first = None
    usage, timings = {}, {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            m = json.loads(line[5:])
            usage = m.get("usage") or usage
            timings = m.get("timings") or timings
            for ch in m.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content")):
                    first = time.time()
    return t0, first or time.time(), time.time(), usage, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--prompt-chars", type=int, default=20000)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="sampling seed (the same text decodes the same way)")
    ap.add_argument("--note", default=None, help="the user message's suffix (default: the time, so no two prompts match)")
    ap.add_argument("--watch", type=float, default=0, help="no request: sample for this many seconds")
    a = ap.parse_args()
    nv = Nvml()
    s = Sampler(nv)
    s.start()
    if a.watch:
        t0 = time.time()
        time.sleep(a.watch)
        phases = [("watch", t0, time.time())]
        info = ""
    else:
        time.sleep(0.5)
        t0, t1, t2, usage, tm = request(a.url, a.prompt_chars, a.max_tokens, a.think, a.seed, a.note)
        phases = [("prompt read", t0, t1), ("decode", t1, t2)]
        gen = usage.get("completion_tokens", 0)
        info = (f"prompt {usage.get('prompt_tokens')} tokens (reused {tm.get('reused')}) in {t1 - t0:.1f} s; "
                f"{gen} tokens in {t2 - t1:.1f} s = {gen / max(t2 - t1, 1e-9):.1f} tok/s")
    time.sleep(0.5)
    s.stop.set()
    s.join()
    if info:
        print(info)
    print(f"{'phase':12s} {'GPU':>3s} {'busy %':>7s} {'memory %':>9s} {'samples':>8s}")
    for name, p0, p1 in phases:
        for i in range(len(nv.handles)):
            g, n = s.mean(i, Nvml.GPU, p0, p1)
            m, _ = s.mean(i, Nvml.MEM, p0, p1)
            print(f"{name:12s} {i:>3d} {g:7.1f} {m:9.1f} {n:8d}")
    print("busy: share of time a kernel ran (NVML); memory: share of time device memory was read or written")


if __name__ == "__main__":
    main()
