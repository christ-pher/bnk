"""A model directory with some tensors replaced from a patch GGUF (an abliteration patch, for example).

    .venv/bin/python tools/tensor_patch.py --model FIRST_SHARD.gguf --patch PATCH.gguf --out DIR
                                           [--pattern 'blk\\.(\\d+)\\.attn_output_b\\.weight$'] [--layers 11-28]

Each tensor of the patch whose name matches --pattern (the group is the layer) and whose layer is in --layers
replaces the model's tensor of the same name, type and shape; the bytes are written in place, so the GGUF layout
does not change. Shards with nothing to replace are symlinked, as are the other .gguf files beside the model (the
DeepSeek Engram tables), so DIR costs only the rewritten shards. A config names DIR's first shard with
"source": {"path": ...}.

huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF (Strata-deepseek-v4.1-flash-IST-DAS/) ships its 40
attn_output_b tensors (Q5_K, identical in ISTA's IQ2_XS and IQ2_XXS builds). Only layers 11-28 carry the ablation
(a rank-1 change, top singular value 9.6 against a 0.18 floor); every layer also carries the noise of a second
Q5_K rounding (~4% relative), so the default --layers 11-28 keeps the original bytes elsewhere (docs/ABLITERATED.md)."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import gguf
import numpy as np


def layer_range(s: str) -> tuple[int, int]:
    a, _, b = s.partition("-")
    return int(a), int(b or a)


def shards(first: Path) -> list[Path]:
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    return [first.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf") for i in range(1, int(m.group(2)) + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="the model's first shard")
    ap.add_argument("--patch", required=True, help="the GGUF holding the replacement tensors")
    ap.add_argument("--out", required=True, help="the directory to write (symlinks + rewritten shards)")
    ap.add_argument("--pattern", default=r"blk\.(\d+)\.attn_output_b\.weight$")
    ap.add_argument("--layers", default="11-28", help="inclusive layer range to replace")
    a = ap.parse_args()

    first, out = Path(a.model).expanduser(), Path(a.out).expanduser()
    lo, hi = layer_range(a.layers)
    pat = re.compile(a.pattern)
    patch = {}
    for t in gguf.GGUFReader(a.patch).tensors:
        m = pat.search(t.name)
        if m and lo <= int(m.group(1)) <= hi:
            patch[t.name] = t
    if not patch:
        print(f"no tensor of {a.patch} matches {a.pattern} in layers {lo}-{hi}")
        return 1
    print(f"{len(patch)} tensors to replace (layers {lo}-{hi})")

    out.mkdir(parents=True, exist_ok=True)
    files = shards(first)
    done = set()
    for f in files:
        dst = out / f.name
        r = gguf.GGUFReader(str(f))
        hits = [(t, patch[t.name]) for t in r.tensors if t.name in patch]
        del r
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if not hits:
            os.symlink(f.resolve(), dst)
            print(f"  {f.name}: linked")
            continue
        for t, p in hits:
            if t.tensor_type != p.tensor_type or list(t.shape) != list(p.shape) or t.n_bytes != p.n_bytes:
                print(f"  {t.name}: {t.tensor_type.name} {list(t.shape)} in the model, "
                      f"{p.tensor_type.name} {list(p.shape)} in the patch: not replaceable in place")
                return 1
        print(f"  {f.name}: copying ({f.stat().st_size / 1e9:.1f} GB)")
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(f, tmp)
        with open(tmp, "r+b") as w:
            for t, p in hits:
                w.seek(t.data_offset)
                w.write(np.asarray(p.data).view(np.uint8).tobytes())
                done.add(t.name)
        # read back: every replaced tensor equals the patch's
        r = gguf.GGUFReader(str(tmp))
        for t in r.tensors:
            if t.name in patch and not np.array_equal(np.asarray(t.data).view(np.uint8), np.asarray(patch[t.name].data).view(np.uint8)):
                print(f"  {t.name}: read-back differs")
                return 1
        del r
        os.replace(tmp, dst)
        print(f"  {f.name}: {len(hits)} tensors replaced")
    missing = set(patch) - done
    if missing:
        print(f"not in the model: {', '.join(sorted(missing))}")
        return 1
    # the other GGUFs beside the model (the Engram tables)
    for g in sorted(first.parent.glob("*.gguf")):
        if g not in files and re.match(r"(.*)-\d{5}-of-\d{5}\.gguf$", g.name) is None:
            dst = out / g.name
            if not dst.exists():
                os.symlink(g.resolve(), dst)
                print(f"  {g.name}: linked")
    print(f"done: {out / first.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
