#!/usr/bin/env python3
# the decode A/B (decode_ab.sh): per turn tok/s, tok/round, verify ms/round, and whether the ids match across variants
import re, sys

def turns(path):
    out, cur = [], None
    for line in open(path):
        m = re.match(r"decode +(\d+) tokens: +([\d.]+) ms +([\d.]+) tok/s +([\d.]+) tok/round, verify ([\d.]+)", line)
        if m:
            cur = {"n": int(m[1]), "tps": float(m[3]), "tpr": float(m[4]), "ver": float(m[5])}
            out.append(cur)
        elif line.startswith("tokens") and cur is not None:
            cur["ids"] = line.split()[1:]
    return out

a, b = turns(sys.argv[1]), turns(sys.argv[2])
print(f"{'turn':>4} {'A tok/s':>8} {'B tok/s':>8} {'A t/rnd':>8} {'B t/rnd':>8} {'A ver':>6} {'B ver':>6}  ids")
for i, (x, y) in enumerate(zip(a, b)):
    same = x.get("ids") == y.get("ids")
    if not same and x.get("ids") and y.get("ids"):
        k = next((j for j, (p, q) in enumerate(zip(x["ids"], y["ids"])) if p != q), min(len(x["ids"]), len(y["ids"])))
        tag = f"differ from token {k}"
    else:
        tag = "identical" if same else "?"
    print(f"{i:>4} {x['tps']:>8.1f} {y['tps']:>8.1f} {x['tpr']:>8.2f} {y['tpr']:>8.2f} {x['ver']:>6.1f} {y['ver']:>6.1f}  {tag}")
mean = lambda v: sum(v) / len(v)
for k, name in (("tps", "tok/s"), ("tpr", "tok/round"), ("ver", "verify ms/round")):
    print(f"mean {name}: A {mean([t[k] for t in a]):.2f}  B {mean([t[k] for t in b]):.2f}")
print(f"identical outputs: {sum(x.get('ids') == y.get('ids') for x, y in zip(a, b))}/{min(len(a), len(b))}")
