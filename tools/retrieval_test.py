#!/usr/bin/env python3
"""Retrieval near the end of long prompts: the zone the long-prompt skip's keep setting affects.

Each prompt is ~TARGET tokens of real text (this repo's docs and sources, a different stretch per prompt so nothing is
reused from a cache), with six facts ("the access code of <name> is <6 digits>") placed at fixed token distances from
the end of the prompt, and a question asking for all six. Greedy, thinking off. Writes one JSON line per prompt.

  retrieval_test.py gen  OUT.jsonl           build the prompts (tokenizer from the model GGUF)
  retrieval_test.py run  PROMPTS OUT URL     ask a running server, record answers and timings
  retrieval_test.py cmp  A.jsonl B.jsonl ... accuracy per distance, and outputs identical to the first file
"""
import glob, json, random, re, sys, time, urllib.request

MODEL = ("/root/.cache/huggingface/hub/models--ISTA-DASLab--DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/snapshots/"
         "85caaf74bcbc760eb163518043e8a93954347c1b/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf")
TARGET = 10000                                 # tokens of user message
DIST = [150, 500, 1000, 1600, 2300, 6000]      # facts' distance from the end in tokens (6000: control, outside both cones)
N_PROMPTS = 25
NAMES = ["orion", "vega", "talos", "kestrel", "marlin", "sable", "quill", "basalt", "cobalt", "juniper", "lumen", "nimbus",
         "osprey", "pylon", "quartz", "rook", "sierra", "tundra", "umber", "vortex", "willow", "xenon", "yarrow", "zephyr"]


def corpus():
    paras = []
    for pat in ["/opt/engines/bnk/docs/*.md", "/opt/engines/bnk/src/*/*.cpp", "/opt/engines/bnk/src/*/*.cu",
                "/opt/engines/bnk/serve/*.py", "/opt/engines/bnk/tools/*.py"]:
        for f in sorted(glob.glob(pat)):
            text = open(f, errors="replace").read()
            paras += [p for p in re.split(r"\n\s*\n", text) if 40 < len(p) < 3000]
    return paras


def gen(out):
    sys.path.insert(0, "/opt/engines/bnk-skip")
    from serve.tokenizer import Tokenizer
    tok = Tokenizer.from_gguf(MODEL)
    ntok = lambda s: len(tok.encode(s, parse_special=False))
    paras = corpus()
    rng = random.Random(1027)
    with open(out, "w") as fo:
        for k in range(N_PROMPTS):
            names = rng.sample(NAMES, len(DIST))
            facts = [(f"{n}-{rng.randint(10, 99)}", f"{rng.randint(100000, 999999)}") for n in names]
            question = ("\n\nQuestion: the text above contains six access codes. List the access code of each of these, one per "
                        "line as `name: code`, nothing else: " + ", ".join(f[0] for f in facts) + ".")
            # the text from a random stretch of the corpus, read backwards from the end: facts dropped in at their distances
            start = rng.randrange(len(paras))
            tail, used, order = [], ntok(question), sorted(range(len(DIST)), key=lambda i: DIST[i])
            fi, i = 0, start
            while used < TARGET:
                while fi < len(order) and used >= DIST[order[fi]]:
                    j = order[fi]
                    s = f"(Note: the access code of {facts[j][0]} is {facts[j][1]}.)"
                    tail.append(s)
                    used += ntok(s) + 2
                    fi += 1
                p = paras[i % len(paras)]
                i -= 1
                tail.append(p)
                used += ntok(p) + 2
            text = "\n\n".join(reversed(tail)) + question
            ids = tok.encode(text, parse_special=False)
            # the real distance of each fact from the end, in tokens
            dist = []
            for name, code in facts:
                at = text.index(f"access code of {name} is")
                dist.append(len(ids) - ntok(text[:at]))
            fo.write(json.dumps({"k": k, "text": text, "facts": facts, "dist": dist, "tokens": len(ids)}) + "\n")
            print(f"prompt {k}: {len(ids)} tokens, facts at {dist}", flush=True)


def run(prompts, out, url):
    with open(prompts) as fi, open(out, "w") as fo:
        for line in fi:
            p = json.loads(line)
            body = {"model": "x", "messages": [{"role": "user", "content": p["text"]}], "temperature": 0, "max_tokens": 160,
                    "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t0 = time.time()
            r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
            dt = time.time() - t0
            ans = r["choices"][0]["message"].get("content") or ""
            got = [(name in ans and re.search(re.escape(name) + r"\D{0,12}(\d{6})", ans)) for name, _ in p["facts"]]
            ok = [bool(g) and g.group(1) == code for g, (_, code) in zip(got, p["facts"])]
            fo.write(json.dumps({"k": p["k"], "answer": ans, "ok": ok, "dist": p["dist"], "s": dt,
                                 "usage": r.get("usage"), "timings": r.get("timings")}) + "\n")
            fo.flush()
            print(f"prompt {p['k']}: {sum(ok)}/{len(ok)} right, {dt:.1f} s", flush=True)


def cmp(files):
    runs = [[json.loads(l) for l in open(f)] for f in files]
    for f, rs in zip(files, runs):
        n = len(DIST)
        per = [sum(r["ok"][i] for r in rs) for i in range(n)]
        print(f"{f}: {sum(per)}/{n * len(rs)} right; by distance " +
              ", ".join(f"~{DIST[i]}: {per[i]}/{len(rs)}" for i in range(n)) +
              f"; mean {sum(r['s'] for r in rs) / len(rs):.1f} s a prompt")
    for f, rs in zip(files[1:], runs[1:]):
        same = sum(a["answer"] == b["answer"] for a, b in zip(runs[0], rs))
        print(f"{f}: answers identical to {files[0]}: {same}/{len(rs)}")


if __name__ == "__main__":
    {"gen": lambda: gen(sys.argv[2]), "run": lambda: run(sys.argv[2], sys.argv[3], sys.argv[4]),
     "cmp": lambda: cmp(sys.argv[2:])}[sys.argv[1]]()
