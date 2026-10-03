"""Agent-style benchmark against a running bnk server: a large code context (this repository's sources), then
short turns that each add a tool result and a question, reusing the conversation so far - the shape of real
agent traffic (long context, ~97% of each prompt shared with the previous turn, generation-dominated).

    .venv/bin/python tools/agent_bench.py [context_chars=220000] [turns=5] [temperature]

Prints per-turn prompt-processing time, decode speed, tokens per round and expert misses, then a summary."""
import glob, json, sys, time, urllib.request
URL = "http://localhost:8080/v1/chat/completions"
target_chars = int(sys.argv[1]) if len(sys.argv) > 1 else 220_000   # ~3.6 chars per token
turns = int(sys.argv[2]) if len(sys.argv) > 2 else 5
files = sorted(glob.glob("/opt/engines/bnk/src/**/*.*", recursive=True)) + sorted(glob.glob("/opt/engines/bnk/serve/*.py"))
ctx, n = [], 0
for f in files:
    t = open(f, errors="replace").read()
    ctx.append(f"=== {f.removeprefix('/opt/engines/bnk/')} ===\n{t}\n")
    n += len(ctx[-1])
    if n > target_chars:
        break
system = "You are a coding agent working in the repository below. Answer precisely, citing files.\n\n" + "".join(ctx)
questions = [
    "Summarize how the expert cache decides which experts to keep in VRAM. Which functions are involved?",
    "Is there any risk of a race between ExpertCache::adapt and the MoE kernels reading slot_of? Explain.",
    "Write a small Python helper that parses the engine's telemetry JSON line and prints decode tok/s.",
    "How does the prefill path decide between grouped GEMMs, list kernels and the CPU for an expert?",
    "Suggest two concrete optimizations for decode speed at long context and say which files you would change.",
    "Explain the elastic VRAM budget in two paragraphs.",
]
msgs = [{"role": "system", "content": system}]
out = []
for i in range(turns + 1):
    q = questions[i % len(questions)] if i else "Read the repository. Reply with one sentence when ready."
    if i:  # like an agent's tool result: a file it just read
        extra_files = sorted(glob.glob("/opt/engines/bnk/tools/*.py")) + sorted(glob.glob("/opt/engines/bnk/serve/web/src/pages/*.tsx"))
        snippet = open(extra_files[i % len(extra_files)], errors="replace").read()[:2000 + 400 * i]
        q = "Tool result (file contents):\n" + snippet + "\n\n" + q
    msgs.append({"role": "user", "content": q})
    body = {"messages": msgs, "max_tokens": 900 if i else 60, "seed": 7 + i}
    if len(sys.argv) > 3: body["temperature"] = float(sys.argv[3])
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=3600).read())
    tm, u = r["timings"], r["usage"]
    msg = r["choices"][0]["message"]
    msgs.append({"role": "assistant", "content": msg.get("content") or ""})
    row = dict(turn=i, ctx=u["prompt_tokens"], new=u["prompt_tokens"] - (tm.get("reused") or 0), prefill_s=(tm.get("prefill_ms") or 0) / 1000,
               gen=u["completion_tokens"], tps=tm.get("tps") or 0, tpr=tm.get("tokens_per_round") or 0, miss=tm.get("expert_miss_rate") or 0,
               cpu_ms=(tm.get("cpu_expert_ms") or 0) / max(1, tm.get("rounds") or 1), verify=(tm.get("verify_ms") or 0) / max(1, tm.get("rounds") or 1))
    out.append(row)
    print(f"turn {i}: ctx {row['ctx']:6d} new {row['new']:6d} read {row['prefill_s']:6.1f} s | gen {row['gen']:4d} at {row['tps']:5.1f} tok/s"
          f" {row['tpr']:.2f} tok/round, miss {row['miss']:.1%}, verify {row['verify']:.0f} ms (cpu {row['cpu_ms']:.0f} ms)/round", flush=True)
dec = [r for r in out if r["turn"] > 0]
gen = sum(r["gen"] for r in dec); secs = sum(r["gen"] / r["tps"] for r in dec if r["tps"])
print(f"SUMMARY decode {gen / secs:.1f} tok/s over {gen} tokens; turn reads {sum(r['prefill_s'] for r in dec) / len(dec):.2f} s avg; "
      f"miss {sum(r['miss'] for r in dec) / len(dec):.1%}")
