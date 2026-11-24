"""Exactness checks for the DeepSeek engine's snapshots, parking and disk states (docs/DS-PARK-PLAN.md).

    .venv/bin/python tools/ds_park_check.py MODEL-00001-of-0000N.gguf [--gpus all] [--cases window,extend,rerender,disk]

Runs `bnk serve` greedily (static expert placement BNK_DS_ADAPT=0, no drafts BNK_DS_LOOKUP=0, BNK_EXACT=1, the
prompt path's CPU / PCIe split pinned by BNK_CPU_US_PER_EXPERT_TOKEN). The engine is still not bit-deterministic
from run to run here (the same cold prompt can decode differently twice), so tokens alone cannot judge a resumed
state. Each case compares the first logits row of the turn under test (BNK_DUMP_LOGITS) against a reference run and
against the noise between two reference runs; a state that misses something shows as a difference far above it.
  window   : turn 2 rolls back to the snapshot at turn 1's user message after a long reply (which overwrites the
             attention window's ring), vs turn 2 read from scratch in the same segments
  extend   : A, B (A parked), A's next turn extending its history, vs A then its next turn
  rerender : the same with A's answer re-rendered differently (A comes back from its snapshot)
  disk     : A, the engine restarted (A written to disk at quit), A's next turn, vs A then its next turn"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve.chat import ChatTemplate  # noqa: E402
from serve.tokenizer import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--engine", default=str(ROOT / "build" / "bnk"))
ap.add_argument("--cases", default="window,extend,rerender,disk")
ap.add_argument("--gpus", default="", help="engine --gpus (all: the two-stage pipeline)")
ap.add_argument("--gen", type=int, default=24)
ap.add_argument("--refs", type=int, default=3, help="reference runs a case (their spread is the noise)")
ap.add_argument("--first", type=int, default=8, help="turn 1's reply length (short: less room for run-to-run drift)")
ap.add_argument("--long", type=int, default=300, help="turn 1's reply length in the window case")
ap.add_argument("--ctx-chars", type=int, default=12000)
ap.add_argument("--park-dir", default=str(ROOT / "build" / "park-check"))
ap.add_argument("--log", default="", help="the engine's stderr (default: inherited)")
a = ap.parse_args()

tok = Tokenizer.from_gguf(a.model, cache_dir=str(Path("~/.cache/bnk").expanduser()))
tmpl = ChatTemplate((ROOT / "templates" / "deepseek-v4.1.jinja").read_text())
user_id = tok.id_of("<｜User｜>")
stops = [t for t in (tok.id_of("<｜end▁of▁sentence｜>"), tok.ids.get("eos_token_id")) if t is not None]
dump = Path(a.park_dir) / "logits.f32"
Path(a.park_dir).mkdir(parents=True, exist_ok=True)
dump.unlink(missing_ok=True)
env = dict(os.environ, BNK_EXACT="1", BNK_DS_ADAPT="0", BNK_DS_LOOKUP="0", BNK_CPU_US_PER_EXPERT_TOKEN="1e9",
           BNK_PARK_DIR=a.park_dir, BNK_DUMP_LOGITS=str(dump))
log = open(a.log, "ab") if a.log else None
eng = None
n = 0


def start():
    global eng
    args = [a.engine, "serve", "--model", a.model, "--ctx", "65536", "--slots", "1", "--park-gib", "8", "--park-min", "0"]
    if a.gpus:
        args += ["--gpus", a.gpus]
    eng = subprocess.Popen(args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or sys.stderr, bufsize=0)
    while read().get("type") != "ready":
        pass


def stop():
    eng.stdin.write(b'{"op":"quit"}\n')
    eng.wait(timeout=300)


def read():
    while True:
        line = eng.stdout.readline()
        if not line:
            raise SystemExit("the engine exited")
        m = json.loads(line)
        if m.get("type") != "telemetry":
            return m


def gen(ids, cps, max_tokens):
    """Greedy request with snapshots at `cps`; returns (generated ids, tokens reused, first logits row)."""
    global n
    n += 1
    before = dump.stat().st_size if dump.exists() else 0
    req = {"op": "generate", "id": f"r{n}", "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "stop_ids": stops,
           "checkpoints": cps}
    eng.stdin.write((json.dumps(req) + "\n").encode())
    out, reused = [], None
    while True:
        m = read()
        if m["type"] == "prefill":
            reused = m["reused"]
        elif m["type"] == "tokens":
            out += m["ids"]
        elif m["type"] == "done":
            break
        elif m["type"] == "error":
            raise SystemExit(m)
    with open(dump, "rb") as f:
        f.seek(before)
        row = np.frombuffer(f.read(), dtype=np.float32)
    return out, reused, row


def reset():
    eng.stdin.write(b'{"op":"reset","id":"x"}\n')
    while read().get("type") != "reset":
        pass


def chat(msgs):
    return tok.encode(tmpl.render(msgs, add_generation_prompt=True, thinking_mode="chat"))


def last_user(ids):
    return len(ids) - 1 - ids[::-1].index(user_id)


src = sorted(ROOT.glob("src/**/*.cpp"))
ctx_a = "".join(p.read_text(errors="replace") for p in src[:6])[: a.ctx_chars]
ctx_b = "".join(p.read_text(errors="replace") for p in src[6:])[: a.ctx_chars]
sys_a = {"role": "system", "content": "You review C++ code.\n\n" + ctx_a}
q_a = {"role": "user", "content": "What does the first file do? Explain it in detail, function by function."}
pa = chat([sys_a, q_a])
pb = chat([{"role": "system", "content": "You answer questions about this code.\n\n" + ctx_b},
           {"role": "user", "content": "List three function names."}])
cpa = last_user(pa)
rer = chat([sys_a, q_a, {"role": "assistant", "content": "It reviews code."}, {"role": "user", "content": "More?"}])
ok = True


def logp(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def kl(p_logits, q_logits):
    lp, lq = logp(p_logits), logp(q_logits)
    return float((np.exp(lp) * (lp - lq)).sum())


def report(case, refs, test, want_reused):
    """refs / test: (tokens, reused, logits). Passes when the test's KL from the references is within the spread
    of the references' KL from each other (the run-to-run noise)."""
    global ok
    noise = max(kl(refs[i][2], refs[j][2]) for i in range(len(refs)) for j in range(len(refs)) if i != j)
    d = float(np.median([kl(r[2], test[2]) for r in refs]))
    tops = {int(np.argmax(r[2])) for r in refs}
    top = int(np.argmax(test[2])) in tops
    good = top and test[1] == want_reused and d <= max(2 * noise, 1e-4)
    print(f"{case:9s}: reused {test[1]} (want {want_reused}); first-token KL vs references {d:.2e} (median), noise "
          f"{noise:.2e} (max of {len(refs)} runs); top-1 {'among the references' if top else 'DIFFERENT'}; tokens "
          f"{'identical' if refs[0][0] == test[0] else 'differ'} -> {'ok' if good else 'BAD'}", flush=True)
    if not good:
        ok = False
        print("  reference:", tok.decode(refs[0][0])[:200].replace("\n", " "))
        print("  test     :", tok.decode(test[0])[:200].replace("\n", " "))


cases = a.cases.split(",")
start()
if "window" in cases:
    cps = [cpa, last_user(rer)]
    refs = []
    for _ in range(a.refs):
        reset()
        refs.append(gen(rer, cps, a.gen))
    reset()
    first = gen(pa, [cpa], a.long)
    test = gen(rer, cps, a.gen)
    report("window", refs, test, cpa)
for case in ("extend", "rerender", "disk"):
    if case not in cases:
        continue
    reset()
    a1 = gen(pa, [cpa], a.first)[0]
    nxt = pa + a1 + tok.encode("<｜User｜>And the second?<｜Assistant｜></think>") if case != "rerender" else rer
    refs = [gen(nxt, [last_user(nxt)], a.gen)]
    for _ in range(a.refs - 1):
        reset()
        if gen(pa, [cpa], a.first)[0] != a1:
            print(f"{case}: turn 1 drifted between runs; retrying the case is the fix")
        refs.append(gen(nxt, [last_user(nxt)], a.gen))
    reset()
    if gen(pa, [cpa], a.first)[0] != a1:
        print(f"{case}: turn 1 drifted between runs; retrying the case is the fix")
    if case == "disk":
        stop()
        start()
    else:
        gen(pb, [last_user(pb)], 8)   # another conversation: A is parked
    test = gen(nxt, [last_user(nxt)], a.gen)
    report(case, refs, test, refs[0][1])
stop()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
