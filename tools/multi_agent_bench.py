"""Several agents sharing one bnk server at once (an orchestrator's workers): each has its own large context and
takes short tool-result turns, all running concurrently. Measures what parked conversations and time slices are
for: per-turn prompt reads and the total wall time.

    .venv/bin/python tools/multi_agent_bench.py [agents=3] [context_chars=120000] [turns=4] [max_tokens=600]

Compare a server started with `--park-gib 0 --slice 0` (one conversation at a time, as before) against the
defaults."""
import glob
import json
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8080/v1/chat/completions"
agents = int(sys.argv[1]) if len(sys.argv) > 1 else 3
target_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
turns = int(sys.argv[3]) if len(sys.argv) > 3 else 4
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 600
files = sorted(glob.glob("/opt/engines/bnk/src/**/*.*", recursive=True)) + sorted(glob.glob("/opt/engines/bnk/serve/*.py"))
tasks = ["Find a possible bug and explain it.", "Suggest one simplification.", "Summarize the data flow.",
         "Which function is the slowest and why?", "Write a unit test idea for one function."]
rows, lock = [], threading.Lock()


def agent(k: int):
    # each agent reads a different slice of the repository (a worker given its own files)
    ctx, n = [], 0
    for f in files[k * 7:] + files[:k * 7]:
        t = open(f, errors="replace").read()
        ctx.append(f"=== {f.removeprefix('/opt/engines/bnk/')} ===\n{t}\n")
        n += len(ctx[-1])
        if n > target_chars:
            break
    msgs = [{"role": "system", "content": f"You are worker {k} of a coding agent. Be brief.\n\n" + "".join(ctx)}]
    for i in range(turns):
        q = tasks[(i + k) % len(tasks)]
        if i:
            q = "Tool result: " + open(files[(k * 5 + i) % len(files)], errors="replace").read()[:1500] + "\n\n" + q
        msgs.append({"role": "user", "content": q})
        body = {"messages": msgs, "max_tokens": max_tokens, "seed": 11 + i + 100 * k}
        t0 = time.time()
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
        tm, u = r["timings"], r["usage"]
        msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""})
        row = dict(agent=k, turn=i, ctx=u["prompt_tokens"], reused=tm.get("reused") or 0,
                   read=(tm.get("prefill_ms") or 0) / 1000, gen=u["completion_tokens"], tps=tm.get("tps") or 0,
                   slices=tm.get("slices", 1), wall=time.time() - t0)
        with lock:
            rows.append(row)
            print(f"agent {k} turn {i}: ctx {row['ctx']:6d} reused {row['reused']:6d} read {row['read']:5.1f} s | "
                  f"gen {row['gen']:4d} at {row['tps']:5.1f} tok/s, {row['slices']} slice(s) | turn {row['wall']:6.1f} s",
                  flush=True)


T0 = time.time()
threads = [threading.Thread(target=agent, args=(k,)) for k in range(agents)]
for t in threads:
    t.start()
    time.sleep(0.5)
for t in threads:
    t.join()
total = time.time() - T0
later = [r for r in rows if r["turn"] > 0]
gen = sum(r["gen"] for r in rows)
print(f"SUMMARY {agents} agents x {turns} turns: wall {total:.0f} s, {gen} tokens generated ({gen / total:.1f} tok/s overall); "
      f"reads after the first turn {sum(r['read'] for r in later):.0f} s total, "
      f"{sum(r['read'] for r in later) / max(1, len(later)):.1f} s avg; "
      f"first turns {sum(r['read'] for r in rows if r['turn'] == 0):.0f} s")
