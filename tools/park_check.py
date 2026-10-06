"""Exactness check for parked conversations: a conversation's next turn must produce the same tokens whether or
not another conversation ran on the engine in between (its state parked to host RAM and brought back).

    .venv/bin/python tools/park_check.py MODEL.gguf [--mtp MTP.gguf --draft-vocab F] [--engine build/bnk]

Runs `bnk serve` with a static, CPU-only expert tier (--cache-gib 0 --adapt-every 0, which keeps decoding
deterministic) and greedy sampling, and compares for each case:
  reference : A, then A's next turn
  parked    : A, then B (another conversation: A is parked), then A's next turn (A comes back)
Cases: the next turn extends A's history (full reuse), and the next turn re-renders A's answer differently (reuse
from the snapshot at the start of A's last turn, which is parked along with the rest)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve.tokenizer import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--mtp", default="")
ap.add_argument("--draft-vocab", default="")
ap.add_argument("--engine", default=str(ROOT / "build" / "bnk"))
ap.add_argument("--gen", type=int, default=48)
ap.add_argument("--ctx-chars", type=int, default=12000)
ap.add_argument("--slots", type=int, default=1, help="engine conversation slots (batched decoding when > 1)")
a = ap.parse_args()

tok = Tokenizer.from_gguf(a.model, cache_dir=str(Path("~/.cache/bnk").expanduser()))
args = [a.engine, "serve", "--model", a.model, "--ctx", "32768", "--cache-gib", "0", "--adapt-every", "0",
        "--park-gib", "8", "--park-min", "0", "--slots", str(a.slots)]
if a.mtp:
    args += ["--mtp", a.mtp]
    if a.draft_vocab:
        args += ["--draft-vocab", a.draft_vocab]
eng = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=0)
im_start = tok.id_of("<|im_start|>")
stops = [t for t in (tok.id_of("<|im_end|>"), tok.ids.get("eos_token_id")) if t is not None]


def read():
    while True:
        line = eng.stdout.readline()
        if not line:
            raise SystemExit("the engine exited")
        m = json.loads(line)
        if m.get("type") != "telemetry":
            return m


while read().get("type") != "ready":
    pass
n = 0


def gen(ids):
    """Greedy request; returns (generated ids, tokens reused)."""
    global n
    n += 1
    last = len(ids) - 1 - ids[::-1].index(im_start)
    req = {"op": "generate", "id": f"r{n}", "prompt": ids, "max_tokens": a.gen, "temperature": 0, "stop_ids": stops,
           "checkpoint": last}
    eng.stdin.write((json.dumps(req) + "\n").encode())
    out, reused = [], None
    while True:
        m = read()
        if m["type"] == "prefill":
            reused = m["reused"]
        elif m["type"] == "tokens":
            out += m["ids"]
        elif m["type"] == "done":
            return out, reused
        elif m["type"] == "error":
            raise SystemExit(m)


def reset():
    eng.stdin.write(b'{"op":"reset","id":"x"}\n')
    while read().get("type") != "reset":
        pass


def chat(system, turns):
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    for role, text in turns:
        s += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    return tok.encode(s + "<|im_start|>assistant\n")


src = sorted(ROOT.glob("src/**/*.cpp"))
ctx_a = "".join(p.read_text(errors="replace") for p in src[:6])[: a.ctx_chars]
ctx_b = "".join(p.read_text(errors="replace") for p in src[6:])[: a.ctx_chars]
sys_a, q_a = "You review C++ code.\n\n" + ctx_a, "What does the first file do? Two sentences."
pa = chat(sys_a, [("user", q_a)])
pb = chat("You answer questions about this code.\n\n" + ctx_b, [("user", "List three function names.")])

ok = True
for case in ("extend", "rerender"):
    runs = []
    for parked in (False, True):
        reset()
        a1, _ = gen(pa)
        if parked:
            gen(pb)
        if case == "extend":   # the engine's history (prompt + answer) continues
            nxt = pa + a1 + tok.encode("<|im_end|>\n<|im_start|>user\nAnd the second?<|im_end|>\n<|im_start|>assistant\n")
        else:                  # the answer re-rendered differently: resume from the snapshot at the turn start
            nxt = chat(sys_a, [("user", q_a), ("assistant", "It reviews code."), ("user", "More?")])
        a2, reused = gen(nxt)
        runs.append((a1, a2, reused))
    same = runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1]
    print(f"{case:9s}: reference reused {runs[0][2]:6d}, parked reused {runs[1][2]:6d}, tokens "
          f"{'IDENTICAL' if same else 'DIFFER'} ({len(runs[0][1])} / {len(runs[1][1])})", flush=True)
    if not same or runs[0][2] != runs[1][2] or not runs[1][2]:
        ok = False
        print("  reference:", tok.decode(runs[0][1])[:200].replace("\n", " "))
        print("  parked   :", tok.decode(runs[1][1])[:200].replace("\n", " "))
eng.stdin.write(b'{"op":"quit"}\n')
eng.wait(timeout=60)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
