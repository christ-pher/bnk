"""Minimal GGUF metadata reader (key/value section only; tensor data is the engine's business)."""
from __future__ import annotations

import struct

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_metadata(path: str, want: set[str] | None = None) -> dict:
    """Returns {key: value}; arrays become lists. `want` limits which keys are decoded (others are skipped)."""
    out = {}
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:4] != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        _, _, n_kv = struct.unpack_from("<IQQ", head, 4)
        buf = f.read(64 << 20)  # metadata (vocab included) fits well inside this
        off = 0

        def need(n):
            nonlocal buf
            while off + n > len(buf):
                more = f.read(64 << 20)
                if not more:
                    raise ValueError("truncated GGUF metadata")
                buf += more

        def rd_str():
            nonlocal off
            need(8)
            (n,) = struct.unpack_from("<Q", buf, off)
            off += 8
            need(n)
            s = buf[off:off + n].decode("utf-8", errors="replace")
            off += n
            return s

        def rd_val(t, keep):
            nonlocal off
            if t == 8:
                return rd_str()
            if t == 9:
                need(12)
                at, n = struct.unpack_from("<IQ", buf, off)
                off += 12
                if at == 8:
                    vals = [rd_str() for _ in range(n)] if keep else [rd_str() and None for _ in range(n)]
                    return vals if keep else None
                fmt = _SCALAR[at]
                sz = struct.calcsize(fmt)
                need(sz * n)
                vals = list(struct.unpack_from("<" + fmt[1] * n, buf, off)) if keep else None
                off += sz * n
                return vals
            fmt = _SCALAR[t]
            sz = struct.calcsize(fmt)
            need(sz)
            (v,) = struct.unpack_from(fmt, buf, off)
            off += sz
            return v

        for _ in range(n_kv):
            key = rd_str()
            need(4)
            (t,) = struct.unpack_from("<I", buf, off)
            off += 4
            keep = want is None or key in want
            v = rd_val(t, keep)
            if keep:
                out[key] = v
    return out
