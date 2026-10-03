"""Download a single-file GGUF from Hugging Face, optionally leaving out named tensors.

    .venv/bin/python tools/fetch_gguf.py --repo peasantsmith/CYBER-FROST-3.8-PS-GUFF \
        --file CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf --drop per_layer_token_embd.weight \
        --out /opt/models/cyber-frost/CYBER-FROST-3.8-PS-Q5_K_M-noPLE.gguf

The output is a valid GGUF: the dropped tensors' entries are removed from the header and the remaining data is
packed with the file's alignment. The bytes are fetched with parallel range requests in 256 MiB pieces and the
download resumes from `<out>.state` when interrupted.

Why: CYBER-FROST-3.8 PS-GUFF (112.5 GB) carries the same 28.8 GB PLE table as Orca IQ4_XS, byte for byte, so it is
stored without it (83.7 GB) and the engine reads the table from the Orca file (`--ple-gguf`).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import threading
import time
import urllib.request

CHUNK = 256 << 20
_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def fetch(url: str, a: int, b: int) -> bytes:
    """Bytes [a, b) of url, retried."""
    hdr = {"Range": f"bytes={a}-{b - 1}", "User-Agent": "bnk-fetch-gguf"}
    if os.environ.get("HF_TOKEN"):
        hdr["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    for k in range(20):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
                d = r.read()
            if len(d) == b - a:
                return d
            print(f"  short read at {a} ({len(d)} of {b - a} bytes)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  retry {k + 1} at {a}: {e}", flush=True)
        time.sleep(min(30, 2 ** k))
    raise SystemExit(f"download failed at byte {a}")


class Header:
    """The GGUF header: key/value bytes kept verbatim, tensor-info records located for rewriting."""

    def __init__(self, b: bytes):
        self.b, self.p = b, 0
        if b[:4] != b"GGUF":
            raise SystemExit("not a GGUF file")
        self.p = 8
        n_tensors, n_kv = self.rd("<Q"), self.rd("<Q")
        self.align = 32
        for _ in range(n_kv):
            key = self.str_()
            t = self.rd("<I")
            if key == "general.alignment":
                self.align = struct.unpack_from("<I", b, self.p)[0]
            self.skip(t)
        self.kv_end = self.p
        self.infos = []  # (name, record start, record end, offset field position within record, data offset)
        for _ in range(n_tensors):
            s = self.p
            name = self.str_()
            nd = self.rd("<I")
            self.p += 8 * nd + 4
            offp = self.p
            off = self.rd("<Q")
            self.infos.append((name, s, self.p, offp - s, off))
        self.data_start = (self.p + self.align - 1) // self.align * self.align

    def rd(self, f):
        v = struct.unpack_from(f, self.b, self.p)[0]
        self.p += struct.calcsize(f)
        return v

    def str_(self):
        n = self.rd("<Q")
        s = self.b[self.p:self.p + n].decode()
        self.p += n
        return s

    def skip(self, t):
        if t in _SCALAR:
            self.p += _SCALAR[t]
        elif t == 8:
            n = self.rd("<Q")
            self.p += n
        elif t == 9:
            at, n = self.rd("<I"), self.rd("<Q")
            for _ in range(n):
                self.skip(at)
        else:
            raise SystemExit(f"unknown GGUF value type {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", required=True, help="the GGUF's path in the repository")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", default="", help="comma-separated tensor names to leave out")
    ap.add_argument("--header-mib", type=int, default=64, help="bytes read to parse the header")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.file}"
    drop = {d for d in args.drop.split(",") if d}

    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "bnk-fetch-gguf"})
    total = int(urllib.request.urlopen(head).headers["Content-Length"])
    h = Header(fetch(url, 0, min(total, args.header_mib << 20)))
    missing = drop - {i[0] for i in h.infos}
    if missing:
        raise SystemExit(f"not in the file: {sorted(missing)}")

    # each tensor's stored size: up to the next tensor (or the end of the file), aligned
    offs = sorted(i[4] for i in h.infos) + [total - h.data_start]
    size = {o: offs[k + 1] - o for k, o in enumerate(offs[:-1])}
    keep = sorted((i for i in h.infos if i[0] not in drop), key=lambda i: i[4])
    new_off, cur = {}, 0
    for i in keep:
        new_off[i[0]] = cur
        cur += (size[i[4]] + h.align - 1) // h.align * h.align
    hdr = bytearray(h.b[:h.kv_end])
    struct.pack_into("<Q", hdr, 8, len(keep))
    for name, s, e, offrel, _ in h.infos:
        if name not in drop:
            rec = bytearray(h.b[s:e])
            struct.pack_into("<Q", rec, offrel, new_off[name])
            hdr += rec
    hdr += b"\0" * (-len(hdr) % h.align)
    out_size = len(hdr) + cur
    print(f"{len(h.infos)} -> {len(keep)} tensors, {total / 1e9:.2f} -> {out_size / 1e9:.2f} GB", flush=True)

    jobs = []
    for i in keep:
        src, dst, n = h.data_start + i[4], len(hdr) + new_off[i[0]], size[i[4]]
        for a in range(0, n, CHUNK):
            jobs.append((src + a, dst + a, min(CHUNK, n - a)))
    state = args.out + ".state"
    done = set(json.load(open(state))) if os.path.exists(state) else set()
    fd = os.open(args.out, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != out_size:
        os.ftruncate(fd, out_size)
    os.pwrite(fd, bytes(hdr), 0)
    lock, todo, got, t0 = threading.Lock(), [j for j in range(len(jobs)) if j not in done], [0], time.time()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                j = todo.pop(0)
            s, d, n = jobs[j]
            os.pwrite(fd, fetch(url, s, s + n), d)
            with lock:
                done.add(j)
                got[0] += n
                with open(state + ".tmp", "w") as f:
                    json.dump(sorted(done), f)
                os.replace(state + ".tmp", state)
                print(f"{len(done)}/{len(jobs)} pieces, {got[0] / 1e9:.1f} GB at {got[0] / (time.time() - t0) / 1e6:.0f} MB/s",
                      flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    os.fsync(fd)
    os.close(fd)
    if len(done) == len(jobs):
        os.remove(state)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
