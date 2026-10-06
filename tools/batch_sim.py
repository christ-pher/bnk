"""Would decoding several conversations in one batch pay on this machine? An estimate from a real routing log.

    BNK_ROUTE_LOG=route.bin <run several conversations through the server, e.g. tools/multi_agent_bench.py>
    .venv/bin/python tools/batch_sim.py route.bin [evict_fraction ...]

The log has, per decode forward: the window's rows, every row's routed experts per layer (and whether each was
resident in VRAM), the per-layer GPU timestamps (plan, start of the wait for the CPU, end of it) and the CPU's
expert time per layer. From the single-stream forwards this fits a per-layer cost model:

    layer = pre(T) + max(gpu_experts(distinct resident experts, T), cpu(distinct missed experts, miss tasks))

where pre is attention / DeltaNet / router, gpu_experts the resident experts plus the shared expert, and cpu the
missed experts (the GPU waits for those). The model is first checked against the measured forwards; then each
step of a batched decode is simulated by merging the conversations' verify windows round by round (the union of
their experts per layer), and its tokens per second compared with taking turns as today.

Keeping several conversations' KV in VRAM at once takes bytes from the expert cache: each evict_fraction (e.g.
0.2 = a fifth of the cache's slots) re-runs the batched simulation with that share of the coldest resident experts
(by how often the log saw them) counted as misses."""
from __future__ import annotations

import collections
import struct
import sys

import numpy as np

MAGIC = 0x54554F52
K = 10


def read(path):
    recs = []
    with open(path, "rb") as f:
        while True:
            h = f.read(24)
            if len(h) < 24:
                break
            magic, T, pos0, commit, hsh, L = struct.unpack("<iiiiIi", h)
            assert magic == MAGIC, "bad record"
            t0, t1 = struct.unpack("<dd", f.read(16))
            ts = np.frombuffer(f.read(8 * L * 3), dtype=np.uint64).reshape(L, 3).astype(np.float64) / 1e3  # us
            cpu = np.frombuffer(f.read(4 * L), dtype=np.float32).astype(np.float64)
            ids = np.frombuffer(f.read(2 * L * T * K), dtype=np.int16).reshape(L, T, K)
            recs.append(dict(T=T, pos0=pos0, commit=commit, conv=hsh, L=L, t0=t0, t1=t1, ts=ts, cpu=cpu, ids=ids))
    return recs


def layer_feats(ids_rows):
    """ids_rows [rows][K] of one layer (negative = missed): distinct resident, distinct missed, miss tasks."""
    flat = ids_rows.reshape(-1)
    hit = set(int(e) for e in flat if e >= 0)
    miss = set(int(-e - 1) for e in flat if e < 0)
    return len(hit), len(miss), int((flat < 0).sum())


def fit(X, y):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=np.float64), rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(1e-9, ((y - np.mean(y)) ** 2).sum())
    return coef, r2


def evict(recs, frac, slots_total):
    """A copy of the routing with the coldest resident experts (the cache losing frac of its slots) as misses."""
    if frac <= 0:
        return recs
    freq = collections.Counter()
    for r in recs:
        for il in range(r["L"]):
            for e in r["ids"][il].reshape(-1):
                if e >= 0:
                    freq[(il, int(e))] += 1
    # the cache swaps experts as routing drifts, so the log may see more distinct resident experts than there are
    # slots; losing frac of the slots then loses about frac of them (the coldest), else the unseen ones go first
    drop_n = int(frac * len(freq)) if len(freq) >= slots_total else max(0, int(frac * slots_total) - (slots_total - len(freq)))
    gone = set(k for k, _ in sorted(freq.items(), key=lambda kv: kv[1])[:drop_n])
    out = []
    for r in recs:
        ids = r["ids"].copy()
        for il in range(r["L"]):
            m = np.vectorize(lambda e: (il, int(e)) in gone)(ids[il]) & (ids[il] >= 0)
            ids[il][m] = -ids[il][m] - 1
        out.append({**r, "ids": ids})
    return out, drop_n


def main(path, fracs=(), slots_total=10606):
    recs = [r for r in read(path) if r["commit"] == 0]   # verify windows: the decoding itself
    if not recs:
        sys.exit("no decode forwards in the log")
    L = recs[0]["L"]
    print(f"{len(recs)} verify forwards, {len({r['conv'] for r in recs})} conversations, window rows "
          f"{collections.Counter(r['T'] for r in recs).most_common()}")

    # per-layer samples for the cost model
    pre_x, pre_y, gx, gy, cx, cy = [], [], [], [], [], []
    tail_x, tail_y = [], []
    for r in recs:
        ts, T = r["ts"], r["T"]
        for il in range(L):
            dh, dm, tasks = layer_feats(r["ids"][il])
            if il > 0:
                pre_x.append([T]); pre_y.append(ts[il, 0] - ts[il - 1, 2])
            gpu_part = ts[il, 1] - ts[il, 0]
            gx.append([dh, T]); gy.append(gpu_part)
            if dm:
                cx.append([dm, tasks]); cy.append(r["cpu"][il])
        fwd_us = (r["t1"] - r["t0"]) * 1e3
        tail_x.append([T]); tail_y.append(fwd_us - (ts[L - 1, 2] - ts[0, 0]))   # layer 0's pre, head, launch, sync
    pre_c, pre_r2 = fit(pre_x, pre_y)
    g_c, g_r2 = fit(gx, gy)
    c_c, c_r2 = fit(cx, cy)
    t_c, t_r2 = fit(tail_x, tail_y)
    print(f"model per layer:  pre = {pre_c[0]:.0f} + {pre_c[1]:.1f}*rows us (R2 {pre_r2:.2f})")
    print(f"                  gpu experts = {g_c[0]:.0f} + {g_c[1]:.1f}*distinct + {g_c[2]:.1f}*rows us (R2 {g_r2:.2f})")
    print(f"                  cpu = {c_c[0]:.0f} + {c_c[1]:.1f}*distinct missed + {c_c[2]:.1f}*miss tasks us (R2 {c_r2:.2f})")
    print(f"per forward:      rest = {t_c[0]:.0f} + {t_c[1]:.1f}*rows us (R2 {t_r2:.2f})")
    # the CPU starts when the plan is posted; the GPU waits from ts1: stall = max(0, cpu + lat - gpu_part)
    lat = np.median([max(0.0, (r["ts"][il, 2] - r["ts"][il, 0]) - r["cpu"][il])
                     for r in recs for il in range(L) if r["cpu"][il] > 0 and r["ts"][il, 2] - r["ts"][il, 1] > 5])
    print(f"                  CPU hand-off latency (median, stalled layers) {lat:.0f} us")

    def step_us(ids_by_layer, rows):
        t = t_c[0] + t_c[1] * rows
        for il in range(L):
            dh, dm, tasks = layer_feats(ids_by_layer[il])
            if il > 0:
                t += pre_c[0] + pre_c[1] * rows
            g = g_c[0] + g_c[1] * dh + g_c[2] * rows
            c = (c_c[0] + c_c[1] * dm + c_c[2] * tasks + lat) if dm else 0.0
            t += max(g, c)
        return t

    meas = np.array([(r["t1"] - r["t0"]) * 1e3 for r in recs])
    pred = np.array([step_us(r["ids"], r["T"]) for r in recs])
    err = (pred - meas) / meas
    print(f"model check on {len(recs)} measured forwards: mean {meas.mean() / 1e3:.2f} ms, predicted "
          f"{pred.mean() / 1e3:.2f} ms, per-forward error median {np.median(np.abs(err)):.1%}, bias {err.mean():+.1%}")

    # rounds per conversation: committed tokens from the next window's position, the gap to the next forward
    # (drafting, commit, sampling) from the host clock
    by = collections.defaultdict(list)
    for r in recs:
        by[r["conv"]].append(r)
    rounds = {}
    for cid, rs in by.items():
        out = []
        for a, b in zip(rs, rs[1:]):
            adv = b["pos0"] - a["pos0"]
            gap = (b["t0"] - a["t1"]) * 1e3
            if 0 < adv <= a["T"] and gap < 50e3:   # consecutive rounds of one request (not a turn boundary)
                out.append(dict(r=a, tokens=adv, gap=gap))
        rounds[cid] = out
    gaps = np.array([x["gap"] for v in rounds.values() for x in v])
    print(f"between forwards (drafting, commit, sampling): median {np.median(gaps) / 1e3:.2f} ms")
    convs = sorted(rounds, key=lambda c: -len(rounds[c]))
    n = min(len(rounds[c]) for c in convs)
    print(f"simulating {len(convs)} conversations in lockstep over {n} rounds each")

    seq_meas = seq_pred = 0.0
    tokens = 0
    for c in convs:
        for x in rounds[c][:n]:
            seq_meas += (x["r"]["t1"] - x["r"]["t0"]) * 1e3 + x["gap"]
            seq_pred += step_us(x["r"]["ids"], x["r"]["T"]) + x["gap"]
            tokens += x["tokens"]
    gap_med = np.median(gaps)
    res = {}
    for B in range(2, len(convs) + 1):
        bt_ser = bt_par = 0.0
        btok = 0
        for g0 in range(0, len(convs) - B + 1, B):
            group = convs[g0:g0 + B]
            for i in range(n):
                xs = [rounds[c][i] for c in group]
                rows = sum(x["r"]["T"] for x in xs)
                merged = [np.concatenate([x["r"]["ids"][il] for x in xs]) for il in range(L)]
                s = step_us(merged, rows)
                bt_ser += s + gap_med * B   # each conversation drafts on its own
                bt_par += s + gap_med       # drafting batched as well
                btok += sum(x["tokens"] for x in xs)
        res[B] = (btok, bt_ser, bt_par)
    base = tokens / (seq_pred / 1e6)
    print(f"\ntaking turns (today): {tokens} tokens, measured {tokens / (seq_meas / 1e6):.1f} tok/s, "
          f"model {base:.1f} tok/s")
    for frac in fracs:
        er, dropped = evict(recs, frac, slots_total)
        idx = {id(r): i for i, r in enumerate(recs)}
        rr = {c: [dict(x, r=er[idx[id(x["r"])]]) for x in rounds[c]] for c in convs}
        B = len(convs)
        ser = par = 0.0
        btok = 0
        for i in range(n):
            xs = [rr[c][i] for c in convs]
            rows = sum(x["r"]["T"] for x in xs)
            s = step_us([np.concatenate([x["r"]["ids"][il] for x in xs]) for il in range(L)], rows)
            ser += s + gap_med * B
            par += s + gap_med
            btok += sum(x["tokens"] for x in xs)
        print(f"batch of {B}, expert cache {frac:.0%} smaller ({dropped} used experts lost): "
              f"{btok / (ser / 1e6):.1f}-{btok / (par / 1e6):.1f} tok/s "
              f"({btok / (ser / 1e6) / base - 1:+.0%} to {btok / (par / 1e6) / base - 1:+.0%})")
    for B, (btok, ser, par) in res.items():
        print(f"batch of {B}: {btok / (ser / 1e6):.1f} tok/s with per-conversation drafting "
              f"({btok / (ser / 1e6) / base - 1:+.0%}), {btok / (par / 1e6):.1f} tok/s with batched drafting "
              f"({btok / (par / 1e6) / base - 1:+.0%}); each conversation at "
              f"{btok / B / (ser / 1e6):.1f}-{btok / B / (par / 1e6):.1f} tok/s")
    # where the batched step's time goes
    if len(convs) >= 2:
        B = len(convs)
        hs, ms_ = [], []
        for i in range(n):
            xs = [rounds[c][i] for c in convs[:B]]
            for il in range(L):
                dh, dm, _ = layer_feats(np.concatenate([x["r"]["ids"][il] for x in xs]))
                hs.append(dh)
                ms_.append(dm)
        one_h = [layer_feats(x["r"]["ids"][il])[0] for c in convs for x in rounds[c][:n] for il in range(L)]
        one_m = [layer_feats(x["r"]["ids"][il])[1] for c in convs for x in rounds[c][:n] for il in range(L)]
        print(f"\nper layer, one conversation: {np.mean(one_h):.1f} resident + {np.mean(one_m):.2f} missed distinct "
              f"experts; batch of {B}: {np.mean(hs):.1f} + {np.mean(ms_):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "route.bin", [float(x) for x in sys.argv[2:]])
