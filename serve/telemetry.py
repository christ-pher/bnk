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
    """`nvidia-smi dmon` as a stream: utilization, power, temperature, clocks, PCIe traffic (1 s), for every GPU.
    `latest` sums them up (utilization averaged, temperature the hottest) and lists each one under "gpus"."""

    def __init__(self):
        self.latest: dict = {}
        self.static: dict = {}
        self._per: dict[str, dict] = {}
        if shutil.which("nvidia-smi"):
            threading.Thread(target=self._run, daemon=True).start()
            threading.Thread(target=self._static, daemon=True).start()

    def _static(self):
        try:
            rows = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,clocks.max.sm,pcie.link.gen.max,"
                                   "pcie.link.width.current,driver_version", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.strip().split("\n")
            gpus = []
            for out in rows:
                name, plim, smax, pgen, pwidth, drv = [x.strip() for x in out.split(",")]
                gpus.append({"name": name, "power_limit_w": _num(plim), "sm_clock_max": _num(smax),
                             "pcie_gen": _num(pgen), "pcie_width": _num(pwidth), "driver": drv})
            st = dict(gpus[0])
            if len(gpus) > 1:
                st["name"] = f"{len(gpus)} x {gpus[0]['name']}" if all(g["name"] == gpus[0]["name"] for g in gpus) \
                    else " + ".join(g["name"] for g in gpus)
                st["power_limit_w"] = sum(g["power_limit_w"] or 0 for g in gpus)
            st["count"] = len(gpus)
            st["gpus"] = gpus
            self.static = st
        except Exception:
            pass

    def _publish(self):
        per = [dict(v, index=int(k)) for k, v in sorted(self._per.items(), key=lambda kv: int(kv[0]))]
        if not per:
            return
        def tot(key, f=sum):
            vals = [g[key] for g in per if g.get(key) is not None]
            return f(vals) if vals else None
        self.latest = {"t": time.time(), "util": tot("util", lambda v: sum(v) / len(v)),
                       "mem_util": tot("mem_util", lambda v: sum(v) / len(v)), "power_w": tot("power_w"),
                       "temp_c": tot("temp_c", max), "mem_temp_c": tot("mem_temp_c", max),
                       "sm_clock": tot("sm_clock", lambda v: sum(v) / len(v)), "mem_clock": tot("mem_clock", max),
                       "pcie_rx_mbs": tot("pcie_rx_mbs"), "pcie_tx_mbs": tot("pcie_tx_mbs"),
                       "gpus": per if len(per) > 1 else None}

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
                    idx = v.get("gpu")
                    if idx is None:
                        continue
                    # a new round of rows starts with GPU 0: publish the previous round first
                    if idx in self._per and self._per[idx].get("_seen"):
                        self._publish()
                        for g in self._per.values():
                            g["_seen"] = False
                    self._per[idx] = {"util": _num(v.get("sm")), "mem_util": _num(v.get("mem")),
                                      "power_w": _num(v.get("pwr")), "temp_c": _num(v.get("gtemp")),
                                      "mem_temp_c": _num(v.get("mtemp")), "sm_clock": _num(v.get("pclk")),
                                      "mem_clock": _num(v.get("mclk")), "pcie_rx_mbs": _num(v.get("rxpci")),
                                      "pcie_tx_mbs": _num(v.get("txpci")), "_seen": True}
                    if len(self._per) == 1:
                        self._publish()
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
    PEAK_PROMPT_MIN = 1000

    def __init__(self):
        self.latest: dict = {}
        self.history: collections.deque = collections.deque(maxlen=self.HISTORY)
        self.bus = Broadcast()
        self.gpu = GpuSampler()
        self.cpu = CpuSampler()
        self.cores: list[float] = []
        self._prev: dict | None = None
        self.layer_hist: collections.deque = collections.deque(maxlen=self.HISTORY + 60)  # (t, routed, misses)
        self._last_live = 0.0
        self.lock = threading.Lock()
        # per-request speeds: the server sets `decoding` (conversations generating now, with their token counts)
        self.decoding = None
        self._conv_prev: dict[int, tuple[float, int]] = {}
        self._conv_tps: dict[int, collections.deque] = {}
        # records since the server started: decode (every request together, 5 s of uninterrupted decoding, as the
        # dashboard's decode number) and prompt reads (a whole prompt of PEAK_PROMPT_MIN new tokens or more)
        self.peaks: dict = {}
        self._dec5: collections.deque = collections.deque(maxlen=5)
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
        m["peaks"] = self.peaks
        return m

    def prompt_read(self, new_tokens: int, ms: float):
        """A prompt read just finished (as the request starts generating): it may set the prompt-processing record."""
        tps = new_tokens / (ms / 1000) if ms > 0 else 0
        if new_tokens >= self.PEAK_PROMPT_MIN and tps > (self.peaks.get("prefill") or {}).get("value", 0):
            self.peaks = {**self.peaks, "prefill": {"value": round(tps, 1), "t": time.time(), "tokens": new_tokens}}

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
               # every request in flight together, per second of wall time (gen_tps is per request: requests
               # decoded in one batch each count the whole step)
               "gen_tps_total": d("gen_tokens") / dt if d("gen_tokens") > 0 else None,
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
                 "gen_tps": r.get("gen_tps"), "gen_tps_total": r.get("gen_tps_total"),
                 "active": cur.get("active", 0), "prefill_tps": r.get("prefill_tps"),
                 "gen_tokens": r.get("gen_tokens", 0), "prefill_tokens": r.get("prefill_tokens", 0),
                 "accept": r.get("accept"), "tokens_per_round": r.get("tokens_per_round"),
                 "miss_rate": r.get("miss_rate"), "swaps": r.get("swaps", 0), "busy": r.get("busy", 0.0),
                 "vram_used_mb": cur.get("vram_used_mb"), "ram_used_mb": (cur.get("ram_total_mb", 0) -
                                                                            cur.get("ram_free_mb", 0)),
                 "cpu": round(cpu, 1), "gpu_util": g.get("util"), "gpu_power_w": g.get("power_w"),
                 "gpu_temp_c": g.get("temp_c"), "pcie_rx_mbs": g.get("pcie_rx_mbs"),
                 "pcie_tx_mbs": g.get("pcie_tx_mbs"), "pos": cur.get("pos")}
            s["reqs"] = self._per_request()
            self._dec5.append(r.get("gen_tps_total"))
            if len(self._dec5) == 5 and all(v is not None for v in self._dec5):
                v = sum(self._dec5) / 5
                if v > (self.peaks.get("decode") or {}).get("value", 0):
                    self.peaks = {**self.peaks, "decode": {"value": round(v, 1), "t": s["t"],
                                                           "active": cur.get("active", 0)}}
            # each GPU's own readings when there are several (the fields above sum them up or take the hottest)
            per = g.get("gpus") or []
            if len(per) > 1:
                vram = {d.get("device"): d.get("vram_used_mb") for d in cur.get("gpus") or []}
                s["gpus"] = [{"util": x.get("util"), "power_w": x.get("power_w"), "temp_c": x.get("temp_c"),
                              "sm_clock": x.get("sm_clock"), "pcie_rx_mbs": x.get("pcie_rx_mbs"),
                              "pcie_tx_mbs": x.get("pcie_tx_mbs"), "vram_used_mb": vram.get(x.get("index"))}
                             for x in per]
            self.history.append(s)
            if cur.get("layer_routed") is not None:
                self.layer_hist.append((cur.get("t", 0), cur["layer_routed"], cur["layer_misses"]))
            self.bus.publish("sample", s)

    def _per_request(self) -> list[dict]:
        """Each generating conversation's tokens per second of wall time over the last 5 s (the same window as the
        dashboard's combined rate, so the requests add up to it)."""
        if self.decoding is None:
            return []
        now = time.time()
        out, seen = [], set()
        for cid, label, n in self.decoding():
            seen.add(cid)
            prev = self._conv_prev.get(cid)
            self._conv_prev[cid] = (now, n)
            if prev is None or now <= prev[0]:
                continue
            q = self._conv_tps.setdefault(cid, collections.deque(maxlen=5))
            q.append(max(0, n - prev[1]) / (now - prev[0]))
            out.append({"conv": cid, "label": label, "tps": round(sum(q) / len(q), 2)})
        for cid in list(self._conv_prev):
            if cid not in seen:
                self._conv_prev.pop(cid, None)
                self._conv_tps.pop(cid, None)
        return out

    def layer_window(self, seconds: float | None) -> dict:
        """Per-layer expert uses and CPU misses over the last `seconds` (None: since the engine started)."""
        with self.lock:
            cur = dict(self.latest)
        routed, misses = cur.get("layer_routed") or [], cur.get("layer_misses") or []
        if seconds is None or not self.layer_hist:
            return {"routed": routed, "misses": misses, "seconds": None}
        t0 = cur.get("t", time.time()) - seconds
        base = self.layer_hist[0]
        for snap in self.layer_hist:
            if snap[0] <= t0:
                base = snap
            else:
                break
        return {"routed": [a - b for a, b in zip(routed, base[1])], "misses": [a - b for a, b in zip(misses, base[2])],
                "seconds": cur.get("t", 0) - base[0]}
