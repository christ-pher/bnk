"""Exactness of the server's scheduling: requests served concurrently (batched rounds, decoding between another
request's prompt chunks) must produce exactly the tokens they produce alone.

    .venv/bin/python tools/serve_check.py MODEL.gguf PROMPT_A.txt PROMPT_B.txt [PROMPT_C.txt] [--mtp MTP.gguf ...]

Prompts are token-id files (as for `bnk run`); use ones longer than one prompt chunk (2,048 tokens) so a read is
interrupted. Runs `bnk serve` with a static, CPU-only expert tier (deterministic) and greedy sampling:
  alone      : each request on a freshly reset engine, one at a time
  together   : the first request starts decoding, then the others arrive (their prompts read while it decodes)
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("prompts", nargs="+")
ap.add_argument("--mtp", default="")
ap.add_argument("--draft-vocab", default="")
ap.add_argument("--engine", default=str(ROOT / "build" / "bnk"))
ap.add_argument("--gen", type=int, default=64)
a = ap.parse_args()

prompts = [[int(x) for x in Path(p).read_text().replace(",", " ").split()] for p in a.prompts]
args = [a.engine, "serve", "--model", a.model, "--ctx", "16384", "--cache-gib", "0", "--adapt-every", "0",
        "--slots", str(len(prompts)), "--park-gib", "0"]
if a.mtp:
    args += ["--mtp", a.mtp]
    if a.draft_vocab:
        args += ["--draft-vocab", a.draft_vocab]
eng = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
events: dict[str, list] = {}
order: list = []   # (id, type, n tokens) in arrival order
lock = threading.Lock()
ready = threading.Event()


def reader():
    for line in eng.stdout:
        m = json.loads(line)
        t = m.get("type")
        if t == "ready":
            ready.set()
        elif t in ("tokens", "done", "error", "prefill", "reset"):
            with lock:
                events.setdefault(m.get("id", ""), []).append(m)
                order.append((m.get("id", ""), t, len(m.get("ids", []))))


threading.Thread(target=reader, daemon=True).start()
ready.wait()


def send(o):
    eng.stdin.write((json.dumps(o) + "\n").encode())


def request(rid, ids):
    send({"op": "generate", "id": rid, "prompt": ids, "max_tokens": a.gen, "temperature": 0, "stop_ids": [-1]})


def wait_done(rid):
    while True:
        with lock:
            ev = events.get(rid, [])
            if any(e["type"] in ("done", "error") for e in ev):
                break
        time.sleep(0.05)
    with lock:
        ev = events[rid]
    if ev[-1]["type"] == "error":
        sys.exit(f"{rid}: {ev[-1].get('message')}")
    return [t for e in ev if e["type"] == "tokens" for t in e["ids"]]


def reset():
    send({"op": "reset", "id": "reset"})
    while True:
        with lock:
            if events.pop("reset", None):
                return
        time.sleep(0.05)


alone = []
for i, p in enumerate(prompts):
    reset()
    request(f"alone{i}", p)
    alone.append(wait_done(f"alone{i}"))
reset()
request("together0", prompts[0])
while True:   # the first request is decoding: the others' prompts are read meanwhile
    with lock:
        if any(e["type"] == "tokens" for e in events.get("together0", [])):
            break
    time.sleep(0.05)
with lock:
    mark = len(order)
for i in range(1, len(prompts)):
    request(f"together{i}", prompts[i])
together = [wait_done(f"together{i}") for i in range(len(prompts))]
send({"op": "quit"})
eng.wait(timeout=60)
ok = True
for i in range(len(prompts)):
    n = min(len(alone[i]), len(together[i]))
    same = n > 0 and alone[i][:n] == together[i][:n]
    ok &= same
    print(f"request {i}: alone {len(alone[i])} tokens, together {len(together[i])}, first {n} "
          f"{'IDENTICAL' if same else 'DIFFER'}")
# tokens of request 0 that arrived while request 1's prompt was being read (decoding between its chunks)
during = 0
for rid, t, n in order[mark:]:
    if rid == "together1" and t == "prefill":
        break
    if rid == "together0" and t == "tokens":
        during += n
print(f"request 0 produced {during} tokens while request 1's prompt was read")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
