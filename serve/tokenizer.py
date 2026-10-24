"""Byte-level BPE (GPT-2 family) with the qwen35 pre-tokenizer, built from the model's GGUF metadata.

Encoding runs in the toks library (actual-computer/toks: hf-exact ids, AVX2 kernels, 20-45x this module's Python
BPE on the V100 host) when its wheel is installed and the GGUF's tokenizer family is one serve/tokenizer_json.py
can write as a tokenizer.json; BNK_TOKS=0 keeps the Python path. Both give the same ids (tools/tok_check.py)."""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import pickle

import threading

import regex

from .gguf_meta import read_metadata

RECENT_PROMPTS = 8   # rendered prompts kept for encode_chat (a conversation's next turn shares most of its text)


def _common_chars(a: str, b: str) -> int:
    """Length of the common prefix of two strings (slice compares run in C)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n   # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo

# llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 expression (the tokenizer.json original, with (?i:) expanded)
QWEN35 = (r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
          r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
# llama.cpp's LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM (DeepSeek-V4.1): three expressions applied in turn, each splitting the
# pieces the previous one left (text between matches stays a piece of its own)
JOYAI_LLM = [r"\p{N}{1,3}",
             r"[一-龥぀-ゟ゠-ヿ]+",
             r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]
PRE_SEQUENCES = {"joyai-llm": JOYAI_LLM}


def _split_sequence(text: str, pres: list) -> list[str]:
    pieces = [text]
    for pre in pres:
        nxt = []
        for p in pieces:
            pos = 0
            for m in pre.finditer(p):
                if m.start() > pos:
                    nxt.append(p[pos:m.start()])
                nxt.append(m.group(0))
                pos = m.end()
            if pos < len(p):
                nxt.append(p[pos:])
        pieces = nxt
    return pieces


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], types: list[int], specials: dict[str, int], pre: str = ""):
        self.tokens = tokens
        self.pre_name = pre
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.ranks = {}
        for i, m in enumerate(merges):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = i
        self.byte_enc = bytes_to_unicode()
        self.byte_dec = {v: k for k, v in self.byte_enc.items()}
        self.types = types
        # control (3) and user-defined (4) tokens are matched literally in text
        self.special = {t: i for i, t in enumerate(tokens) if types and types[i] in (3, 4)}
        self.special_re = regex.compile("|".join(regex.escape(s) for s in sorted(self.special, key=len, reverse=True)))
        self.pre = regex.compile(QWEN35)
        self.pre_seq = [regex.compile(r) for r in PRE_SEQUENCES.get(pre, [])]
        self.cache: dict[str, list[int]] = {}
        self.wcache: dict[str, list[int]] = {}
        self.byte_table = {i: c for i, c in self.byte_enc.items()}
        self._recent: list = []
        self._recent_lock = threading.Lock()
        self.ids = specials
        self.token_bytes = [self._bytes_of(i) for i in range(len(tokens))]
        self.toks = None          # a toks.Tokenizer over the same vocabulary, or None (the Python path)
        self.toks_n = 0           # ids the toks table knows: [0, toks_n); the GGUF's padding rows lie beyond
        self.backend = "python"

    # ----------------------------------------------------------------- loading
    @classmethod
    def from_gguf(cls, path: str, cache_dir: str | None = None) -> "Tokenizer":
        st = os.stat(path)
        key = hashlib.sha1(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cp = os.path.join(cache_dir, f"tok-{key}.pkl")
            if os.path.exists(cp):
                with open(cp, "rb") as f:
                    tok = pickle.load(f)
                tok._attach_toks(path, cache_dir, key)
                return tok
        md = read_metadata(path, {"tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
                                  "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
                                  "tokenizer.ggml.bos_token_id", "tokenizer.chat_template", "general.name",
                                  "general.sampling.temp", "general.sampling.top_k", "general.sampling.top_p",
                                  "tokenizer.ggml.pre"})
        specials = {k.split(".")[-1]: v for k, v in md.items() if k.endswith("_token_id")}
        tok = cls(md["tokenizer.ggml.tokens"], md["tokenizer.ggml.merges"], md.get("tokenizer.ggml.token_type") or [],
                  specials, md.get("tokenizer.ggml.pre", ""))
        tok.chat_template = md.get("tokenizer.chat_template", "")
        tok.model_name = md.get("general.name", "model")
        tok.sampling = {k.split(".")[-1]: v for k, v in md.items() if k.startswith("general.sampling.")}
        if cache_dir:
            with open(cp, "wb") as f:
                pickle.dump(tok, f)
        tok._attach_toks(path, cache_dir, key)
        return tok

    def _attach_toks(self, path: str, cache_dir: str | None, key: str) -> None:
        """Loads the toks library over this vocabulary: the GGUF's tokenizer written as a tokenizer.json beside
        the pickle (once), then toks_load. Any failure leaves the Python path."""
        self.toks, self.toks_n, self.backend = None, 0, "python"
        if os.environ.get("BNK_TOKS", "1") == "0":
            return
        try:
            import toks
            from .tokenizer_json import write_tokenizer_json
            jp = os.path.join(cache_dir or os.path.dirname(path), f"tok-{key}.json")
            if not os.path.exists(jp) and not write_tokenizer_json(path, jp):
                return
            tk = toks.Tokenizer.from_file(jp)
            n = tk.info().get("n_ids", 0)
            # the toks table must spell every id as this module does (the GGUF's vocabulary, the same bytes)
            for i in (0, 1, n // 2, n - 1):
                if tk.token_bytes(i) != self.token_bytes[i]:
                    return
            self.toks, self.toks_n = tk, n
            self.backend = f"toks {toks.__version__} ({tk.info().get('tier', '?')})"
        except Exception:
            self.toks, self.toks_n, self.backend = None, 0, "python"

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("special_re", None)
        d.pop("pre", None)
        d.pop("pre_seq", None)
        d.pop("_recent_lock", None)
        d.pop("toks", None)
        d["cache"] = {}
        d["wcache"] = {}
        d["_recent"] = []
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.toks, self.toks_n, self.backend = None, 0, "python"
        self.wcache = {}
        self.byte_table = {i: c for i, c in self.byte_enc.items()}
        self._recent = []
        self._recent_lock = threading.Lock()
        self.special_re = regex.compile("|".join(regex.escape(s) for s in sorted(self.special, key=len, reverse=True)))
        self.pre = regex.compile(QWEN35)
        self.pre_seq = [regex.compile(r) for r in PRE_SEQUENCES.get(getattr(self, "pre_name", ""), [])]

    # ----------------------------------------------------------------- encode
    def _bpe(self, word: str) -> list[int]:
        hit = self.cache.get(word)
        if hit is not None:
            return hit
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi < 0:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        out = []
        for p in parts:
            i = self.vocab.get(p)
            if i is None:  # should not happen with a byte-level vocab: fall back to single bytes
                out.extend(self.vocab[c] for c in p)
            else:
                out.append(i)
        if len(self.cache) < 200000:
            self.cache[word] = out
        return out

    def _encode_plain(self, text: str) -> list[int]:
        if self.toks is not None:
            # a segment between added tokens: hf's pipeline without added-token extraction, no template
            return self.toks.encode(text, add_special_tokens=False, added_tokens="none")
        # the cache is keyed by the word as written: a hit skips the byte mapping too (most words, after the first
        # few requests); the mapping itself is a str.translate over the word's UTF-8 bytes read as Latin-1
        out, wcache, bpe, table = [], self.wcache, self._bpe, self.byte_table
        words = _split_sequence(text, self.pre_seq) if self.pre_seq else self.pre.findall(text)
        for w in words:
            ids = wcache.get(w)
            if ids is None:
                ids = bpe(w.encode("utf-8").decode("latin-1").translate(table))
                if len(wcache) < 200000:
                    wcache[w] = ids
            out += ids
        return out

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        if not parse_special or not self.special:
            return self._encode_plain(text)
        return self._encode_marked(text)[0]

    def _encode_marked(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """The tokens, and (character, token) offsets just past each special token: text is encoded segment by
        segment between special tokens, so the tokens of text[:c] for such a c are exactly the first t tokens."""
        out, pos, marks = [], 0, []
        for m in self.special_re.finditer(text):
            if m.start() > pos:
                out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.special[m.group(0)])
            pos = m.end()
            marks.append((pos, len(out)))
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out, marks

    def encode_chat(self, text: str) -> list[int]:
        """encode(text), reusing a recent prompt's tokens up to the last special token inside the text they share:
        a conversation's next turn re-renders every earlier message, and encoding all of it again cost ~0.5-0.7 s
        per turn at 256K tokens."""
        best, best_c = None, 0
        with self._recent_lock:
            for e in self._recent:
                c = _common_chars(e[0], text)
                if c > best_c:
                    best, best_c = e, c
        start, keep, prev_marks = 0, 0, []
        if best is not None:
            for i in range(len(best[2]) - 1, -1, -1):
                if best[2][i][0] <= best_c:
                    start, keep = best[2][i]
                    prev_marks = best[2][:i + 1]
                    break
        if start:
            ids, marks = self._encode_marked(text[start:])
            ids = best[1][:keep] + ids
            marks = prev_marks + [(c + start, t + keep) for c, t in marks]
        else:
            ids, marks = self._encode_marked(text)
        with self._recent_lock:
            self._recent.insert(0, (text, ids, marks))
            del self._recent[RECENT_PROMPTS:]
        return ids

    # ----------------------------------------------------------------- decode
    def _bytes_of(self, i: int) -> bytes:
        t = self.tokens[i]
        if self.types and self.types[i] in (3, 4):
            return t.encode("utf-8")
        try:
            return bytes(self.byte_dec[c] for c in t)
        except KeyError:
            return t.encode("utf-8")

    def decode(self, ids: list[int]) -> str:
        if self.toks is not None and all(0 <= i < self.toks_n for i in ids):
            return self.toks.decode(ids, skip_special_tokens=False)
        return b"".join(self.token_bytes[i] for i in ids if 0 <= i < len(self.tokens)).decode("utf-8", errors="replace")

    def id_of(self, text: str) -> int | None:
        return self.vocab.get(text)


class StreamDecoder:
    """Incremental UTF-8: a multi-byte character split across tokens is held until complete."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, ids: list[int]) -> str:
        return self.dec.decode(b"".join(self.tok.token_bytes[i] for i in ids))

    def flush(self) -> str:
        return self.dec.decode(b"", final=True)
