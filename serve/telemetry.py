"""Live telemetry: engine snapshots, GPU and CPU samples, a one-hour history at 1 s, and a broadcast for SSE clients.

The engine pushes a `telemetry` line every 250 ms while it works (every second while idle) with cumulative counters;
rates here are differences of two snapshots over their own timestamps, so they are exact whatever the cadence.
"""
from __future__ import annotations

import collections
import queue
import shutil
import subprocess
import threading
import time


class Broadcast:
    """Fan-out of events to subscribers; a slow subscriber loses events rather than stalling the rest."""

    def __init__(self):
        self.subs: set[queue.Queue] = set()
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=512)
        with self.lock:
            self.subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            self.subs.discard(q)

    def publish(self, event: str, data):
        with self.lock:
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait((event, data))
            except queue.Full:
                pass


class GpuSampler:
    """`nvidia-smi dmon` as a stream: utilization, power, temperature, clocks, PCIe traffic (1 s)."""

    def __init__(self):
        self.latest: dict = {}
        self.static: dict = {}
        if shutil.which("nvidia-smi"):
            threading.Thread(target=self._run, daemon=True).start()
            threading.Thread(target=self._static, daemon=True).start()

    def _static(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,clocks.max.sm,pcie.link.gen.max,"
                                  "pcie.link.width.current,driver_version", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0]
            name, plim, smax, pgen, pwidth, drv = [x.strip() for x in out.split(",")]
            self.static = {"name": name, "power_limit_w": _num(plim), "sm_clock_max": _num(smax),
                           "pcie_gen": _num(pgen), "pcie_width": _num(pwidth), "driver": drv}
        except Exception:
            pass

    def _run(self):
        while True:
            try:
                p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "putcm", "-d", "1"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
                cols = []
                for line in p.stdout:
                    if line.startswith("# gpu") or line.startswith("#gpu"):
                        cols = line.lstrip("#").split()
                        continue
                    if line.startswith("#") or not cols:
                        continue
                    v = dict(zip(cols, line.split()))
                    if v.get("gpu") not in ("0", None):
                        continue
                    self.latest = {"t": time.time(), "util": _num(v.get("sm")), "mem_util": _num(v.get("mem")),
                                   "power_w": _num(v.get("pwr")), "temp_c": _num(v.get("gtemp")),
                                   "mem_temp_c": _num(v.get("mtemp")), "sm_clock": _num(v.get("pclk")),
                                   "mem_clock": _num(v.get("mclk")), "pcie_rx_mbs": _num(v.get("rxpci")),
                                   "pcie_tx_mbs": _num(v.get("txpci"))}
            except Exception:
                pass
            time.sleep(5)


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


class CpuSampler:
    """Overall and per-core CPU utilization from /proc/stat."""

    def __init__(self):
        self.prev = self._read()

    @staticmethod
    def _read():
        cores = []
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if not line.startswith("cpu"):
                        break
                    parts = line.split()
                    vals = list(map(int, parts[1:]))
                    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                    cores.append((parts[0], sum(vals), idle))
        except OSError:
            pass
        return cores

    def sample(self):
        cur = self._read()
        out = []
        for (n0, t0, i0), (n1, t1, i1) in zip(self.prev, cur):
            dt = t1 - t0
            out.append(100.0 * (1 - (i1 - i0) / dt) if dt > 0 else 0.0)
        self.prev = cur
        return (out[0] if out else 0.0), out[1:]


class Telemetry:
    HISTORY = 3600  # seconds kept at 1 s

    def __init__(self):
        self.latest: dict = {}
        self.history: collections.deque = collections.deque(maxlen=self.HISTORY)
        self.bus = Broadcast()
        self.gpu = GpuSampler()
        self.cpu = CpuSampler()
        self.cores: list[float] = []
        self._prev: dict | None = None
        self._last_live = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._sampler, daemon=True).start()

    # the engine's telemetry lines arrive here (engine reader thread)
    def on_engine(self, msg: dict):
        with self.lock:
            self.latest = msg
        t = time.time()
        if t - self._last_live >= 0.2 or msg.get("phase") == "idle":
            self._last_live = t
            self.bus.publish("live", self.live())

    def live(self) -> dict:
        with self.lock:
            m = dict(self.latest)
        m["gpu"] = self.gpu.latest
        m["cores"] = self.cores
        return m

    def _rate(self, a: dict, b: dict) -> dict:
        la, lb = a.get("life", {}), b.get("life", {})
        dt = b.get("t", 0) - a.get("t", 0)
        if dt <= 0:
            return {}

        def d(k):
            return lb.get(k, 0) - la.get(k, 0)

        # speeds over the time actually spent in each phase (None when the engine did none of it this second),
        # so a 0.3 s prefill reads at its true rate rather than averaged over the whole second
        gms, pms = d("gen_ms"), d("prefill_ms")
        out = {"gen_tps": d("gen_tokens") / (gms / 1000) if gms > 50 else None,
               "prefill_tps": d("prefill_tokens") / (pms / 1000) if pms > 50 and d("prefill_tokens") > 0 else None,
               "gen_tokens": d("gen_tokens"), "prefill_tokens": d("prefill_tokens")}
        out["accept"] = d("accepted") / d("drafted") if d("drafted") > 0 else None
        out["tokens_per_round"] = d("gen_tokens") / d("rounds") if d("rounds") > 0 else None
        out["miss_rate"] = d("misses") / d("routed") if d("routed") > 0 else None
        out["swaps"] = d("swaps")
        out["busy"] = (d("gen_ms") + d("prefill_ms")) / (dt * 1000)
        return out

    def _sampler(self):
        while True:
            time.sleep(1.0 - (time.time() % 1.0))
            cpu, cores = self.cpu.sample()
            self.cores = [round(c, 1) for c in cores]
            with self.lock:
                cur = dict(self.latest)
            if not cur:
                continue
            r = self._rate(self._prev, cur) if self._prev else {}
            self._prev = cur
            g = self.gpu.latest
            s = {"t": round(time.time(), 3), "phase": cur.get("phase"),
                 "gen_tps": r.get("gen_tps"), "prefill_tps": r.get("prefill_tps"),
                 "gen_tokens": r.get("gen_tokens", 0), "prefill_tokens": r.get("prefill_tokens", 0),
                 "accept": r.get("accept"), "tokens_per_round": r.get("tokens_per_round"),
                 "miss_rate": r.get("miss_rate"), "swaps": r.get("swaps", 0), "busy": r.get("busy", 0.0),
                 "vram_used_mb": cur.get("vram_used_mb"), "ram_used_mb": (cur.get("ram_total_mb", 0) -
                                                                            cur.get("ram_free_mb", 0)),
                 "cpu": round(cpu, 1), "gpu_util": g.get("util"), "gpu_power_w": g.get("power_w"),
                 "gpu_temp_c": g.get("temp_c"), "pcie_rx_mbs": g.get("pcie_rx_mbs"),
                 "pcie_tx_mbs": g.get("pcie_tx_mbs"), "pos": cur.get("pos")}
            self.history.append(s)
            self.bus.publish("sample", s)
