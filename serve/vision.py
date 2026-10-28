"""Images in chat requests: found in the messages (OpenAI image_url parts, Anthropic image blocks), decoded,
resized for the vision encoder (Qwen2-VL's smart_resize: sides multiples of 32, the pixel count within the
token limits) and written as raw RGB files the engine reads when it encodes them. In the prompt an image is a
run of placeholder tokens, one per merged 32x32 block, all holding the image's id: -1 - (a hash of its pixels),
so the engine's prefix reuse tells images apart like tokens (src/vision/images.h).

DeepSeek-V4.1 (projector deepseek41v) sizes images its own way (deepseek_grid: 14-pixel patches, one cell per 3x3 of
them, at most 1,024 cells, grey letterboxing) and reads an image as a span of cells: a start cell, each row of the
grid followed by a newline cell, an end cell; the run of placeholder ids covers the whole span."""
from __future__ import annotations

import base64
import binascii
import collections
import hashlib
import io
import math
import os
import threading
import urllib.request

try:
    from PIL import Image
except ImportError:   # text-only installs: a request with images gets a clear error
    Image = None


class ImageError(ValueError):
    pass


def smart_resize(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL's sizing: both sides multiples of `factor`, the area within [min_pixels, max_pixels], the aspect
    ratio kept as closely as that allows."""
    if max(h, w) / max(1, min(h, w)) > 200:
        raise ImageError(f"image aspect ratio {max(h, w) / min(h, w):.0f}:1 is too extreme")
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def deepseek_grid(w: int, h: int, patch: int = 14, merge: int = 3, max_cells: int = 1024,
                  min_pixels: int = 544 * 544) -> tuple[int, int, int, int]:
    """DeepSeek-V4.1's plan for a w x h image (inference/image_processor.py plan_image_grid, without a width/height
    ratio cap): (grid rows, grid columns, padded height, padded width). Small images grow to min_pixels; the span
    (rows * (columns + 1) + 2 cells) stays within max_cells."""
    if 0 < w * h < min_pixels:
        r = (min_pixels / (w * h)) ** 0.5
        w, h = int(w * r), int(h * r)
    bw, bh = math.ceil(w / patch) * patch, math.ceil(h / patch) * patch

    def grid(bh_, bw_):
        return math.ceil((bh_ // patch) / merge), math.ceil((bw_ // patch) / merge)

    gh, gw = grid(bh, bw)
    if gh * (gw + 1) + 2 > max_cells:
        ratio = h / w
        mw = math.sqrt((max_cells - 2) / ratio + 0.25) - 0.5
        mh = mw * ratio
        cell = patch * merge
        if mw < 1.0:     # very tall: one column
            bh, bw = (max_cells - 2) // 2 * cell, cell
        elif mh < 1.0:   # very wide: one row
            bh, bw = cell, (max_cells - 3) * cell
        else:
            beta = min(math.floor(mw) * cell / w, math.floor(mh) * cell / h)
            bh, bw = math.floor(h * beta / patch) * patch, math.floor(w * beta / patch) * patch
        gh, gw = grid(bh, bw)
        if gh * (gw + 1) + 2 > max_cells:
            raise ImageError(f"cannot fit a {w}x{h} image in {max_cells} cells")
    return gh, gw, bh, bw


def call_order(messages) -> list:
    """DeepSeek-V4.1 renders parallel tool results in the order of the calls (encoding.py
    sort_tool_results_by_call_order): the messages with each run of user and tool messages' tool results put in that
    order, so that the images are listed in the order the template renders them."""
    out, order, i = list(messages or []), {}, 0
    while i < len(out):
        m = out[i]
        role = m.get("role") if isinstance(m, dict) else None
        if role == "assistant" and m.get("tool_calls"):
            order = {}
            for k, tc in enumerate(m["tool_calls"]):
                tid = tc.get("id") or (tc.get("function") or {}).get("id", "")
                if tid:
                    order[tid] = k
        if role not in ("user", "tool"):
            i += 1
            continue
        j = i
        while j < len(out) and isinstance(out[j], dict) and out[j].get("role") in ("user", "tool"):
            j += 1
        slots = [k for k in range(i, j) if out[k].get("role") == "tool"]
        if len(slots) > 1 and order:
            tools = sorted((out[k] for k in slots), key=lambda t: order.get(t.get("tool_call_id", ""), 0))
            for k, t in zip(slots, tools):
                out[k] = t
        i = j
    return out


def image_parts(messages) -> list:
    """The image sources of OpenAI-style messages in prompt order (as the chat template renders them): each an
    `image_url` URL string or an `image` part's source."""
    out = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "image_url" or "image_url" in p:
                u = p.get("image_url")
                out.append(u.get("url") if isinstance(u, dict) else u)
            elif p.get("type") == "image" or "image" in p:
                out.append(p.get("image") or p.get("source") or p.get("url"))
    return out


def anthropic_image(block: dict) -> dict:
    """An Anthropic image block as an OpenAI image_url part."""
    src = block.get("source") or {}
    if src.get("type") == "base64":
        return {"type": "image_url", "image_url": {"url": f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"}}
    if src.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": src.get("url", "")}}
    raise ImageError(f"unsupported image source type {src.get('type')!r}")


class Images:
    """Prepares images for the engine. Files live in `dir` (kept under `max_dir_bytes`, least recently used
    removed first); a recent source's preparation is remembered, so a conversation resending its images every
    turn decodes each once."""

    def __init__(self, dir: str, unit: int = 32, min_tokens: int = 8, max_tokens: int = 4096,
                 max_dir_bytes: int = 4 << 30, fetch_urls: bool = True, max_fetch_bytes: int = 32 << 20,
                 deepseek: dict | None = None):
        self.dir = dir
        self.deepseek = deepseek   # DeepSeek's grid: {"patch", "merge", "min_pixels"}; max_tokens caps its span
        self.unit, self.min_tokens, self.max_tokens = unit, min_tokens, max_tokens
        self.max_dir_bytes = max_dir_bytes
        self.fetch_urls, self.max_fetch_bytes = fetch_urls, max_fetch_bytes
        os.makedirs(dir, exist_ok=True)
        self._recent: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._written = 0   # bytes written since the last sweep

    # ---- sources
    def _bytes(self, src) -> bytes:
        if isinstance(src, dict):   # an Anthropic-style source passed through
            return self._bytes(anthropic_image({"source": src})["image_url"]["url"])
        if not isinstance(src, str) or not src:
            raise ImageError("image without a URL or data")
        if src.startswith("data:"):
            head, _, data = src.partition(",")
            if ";base64" not in head:
                raise ImageError("only base64 data URLs are supported")
            try:
                return base64.b64decode(data, validate=False)
            except (binascii.Error, ValueError) as e:
                raise ImageError(f"bad base64 image data: {e}") from None
        if src.startswith(("http://", "https://")):
            if not self.fetch_urls:
                raise ImageError("image URLs are disabled (send the image as a data URL)")
            try:
                with urllib.request.urlopen(urllib.request.Request(src, headers={"User-Agent": "bnk"}), timeout=20) as r:
                    data = r.read(self.max_fetch_bytes + 1)
            except Exception as e:
                raise ImageError(f"cannot fetch the image {src[:200]}: {e}") from None
            if len(data) > self.max_fetch_bytes:
                raise ImageError(f"the image at {src[:200]} is larger than {self.max_fetch_bytes >> 20} MiB")
            return data
        try:   # bare base64 (some clients)
            return base64.b64decode(src, validate=True)
        except (binascii.Error, ValueError):
            raise ImageError(f"unsupported image reference {src[:80]!r} (data URL or http(s) URL)") from None

    def prepare(self, src) -> dict:
        """-> {"id", "path", "h", "w", "tokens"} for one image source."""
        if Image is None:
            raise ImageError("images need Pillow (pip install pillow)")
        raw = self._bytes(src)
        key = hashlib.sha256(raw).hexdigest()
        with self._lock:
            got = self._recent.get(key)
            if got and os.path.exists(got["path"]):
                self._recent.move_to_end(key)
                return got
        try:
            im = Image.open(io.BytesIO(raw))
            im.load()
        except Exception as e:
            raise ImageError(f"cannot decode the image: {e}") from None
        if im.mode in ("RGBA", "LA", "P"):   # transparency over white (screenshots, icons)
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        if self.deepseek:
            d = self.deepseek
            gh, gw, h, w = deepseek_grid(im.width, im.height, d["patch"], d["merge"], self.max_tokens, d["min_pixels"])
            if (h, w) != (im.height, im.width):   # letterboxed in grey, as the reference (ImageOps.pad)
                from PIL import ImageOps
                im = ImageOps.pad(im, (w, h), color=(127, 127, 127))
            cells = gh * (gw + 1) + 2
        else:
            u = self.unit
            h, w = smart_resize(im.height, im.width, u, self.min_tokens * u * u, self.max_tokens * u * u)
            if (h, w) != (im.height, im.width):
                im = im.resize((w, h), Image.BICUBIC)
            cells = (h // u) * (w // u)
        pix = im.tobytes()
        digest = hashlib.sha256(pix + f"{w}x{h}".encode()).digest()
        ident = -1 - (int.from_bytes(digest[:4], "little") & 0x3FFFFFFF)
        path = os.path.join(self.dir, f"{digest[:8].hex()}-{w}x{h}.rgb")
        if os.path.exists(path):
            os.utime(path)
        else:
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(pix)
            os.replace(tmp, path)
            self._written += len(pix)
        got = {"id": ident, "path": path, "h": h, "w": w, "tokens": cells}
        with self._lock:
            self._recent[key] = got
            while len(self._recent) > 256:
                self._recent.popitem(last=False)
            if self._written > self.max_dir_bytes // 8:
                self._written = 0
                self._sweep()
        return got

    def _sweep(self):
        files = []
        for n in os.listdir(self.dir):
            p = os.path.join(self.dir, n)
            try:
                st = os.stat(p)
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, p))
        total = sum(f[1] for f in files)
        for _, size, p in sorted(files):
            if total <= self.max_dir_bytes:
                break
            try:
                os.remove(p)
                total -= size
            except OSError:
                pass

    # ---- prompts
    def expand(self, ids: list[int], pad_id: int, messages) -> tuple[list[int], list[dict]]:
        """The rendered prompt's tokens with every image pad token (one per image, in prompt order) replaced by
        the image's run of placeholder ids, and the images the engine needs."""
        srcs = image_parts(messages)
        n_pad = sum(1 for t in ids if t == pad_id)
        if not srcs and not n_pad:
            return ids, []
        if n_pad != len(srcs):
            raise ImageError(f"the prompt has {n_pad} image placeholders for {len(srcs)} images")
        imgs = [self.prepare(s) for s in srcs]
        out, k = [], 0
        for t in ids:
            if t == pad_id:
                out.extend([imgs[k]["id"]] * imgs[k]["tokens"])
                k += 1
            else:
                out.append(t)
        uniq = {i["id"]: i for i in imgs}
        return out, [{"id": i["id"], "path": i["path"], "h": i["h"], "w": i["w"]} for i in uniq.values()]
