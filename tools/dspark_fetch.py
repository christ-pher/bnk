"""Fetch only the DSpark (mtp.*) tensors of DeepSeek-V4.1-Flash-1017 from the Hugging Face safetensors shards with HTTP
range requests: each tensor to <out>/<name>.bin, plus manifest.json {name: {dtype, shape, shard, bytes}}."""
import json, os, struct, sys, time, urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def get(url, rng=None, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url)
            if rng: req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(2 * (t + 1))

idx = json.loads(get(BASE + "model.safetensors.index.json"))["weight_map"]
shards = sorted({v for k, v in idx.items() if k.startswith("mtp.")})
manifest = {}
mpath = os.path.join(out, "manifest.json")
if os.path.exists(mpath): manifest = json.load(open(mpath))
total = done = 0
for sh in shards:
    n = struct.unpack("<Q", get(BASE + sh, (0, 7)))[0]
    hdr = json.loads(get(BASE + sh, (8, 8 + n - 1)))
    base = 8 + n
    for name, meta in hdr.items():
        if not name.startswith("mtp."): continue
        a, b = meta["data_offsets"]
        total += b - a
        fn = os.path.join(out, name + ".bin")
        if name in manifest and os.path.exists(fn) and os.path.getsize(fn) == b - a:
            done += b - a; continue
        data = get(BASE + sh, (base + a, base + b - 1))
        assert len(data) == b - a, (name, len(data), b - a)
        with open(fn + ".part", "wb") as f: f.write(data)
        os.replace(fn + ".part", fn)
        manifest[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "shard": sh, "bytes": b - a}
        json.dump(manifest, open(mpath, "w"), indent=0)
        done += b - a
        print(f"{done / 1e9:7.2f} / {total / 1e9:.2f} GB  {name}", flush=True)
print(f"done: {len(manifest)} tensors, {total / 1e9:.2f} GB")
