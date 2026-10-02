"""The drafter's token subset: the vocabulary rows the MTP draft head scores (int32 ids, ascending).

    .venv/bin/python tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf --out ~/.cache/bnk/draft-vocab.bin

A token is kept when it is a special/control token, a single-byte fallback token, or its text is made only of the
scripts and symbol blocks below (Latin, punctuation, digits, math, technical symbols, box drawing, emoji). The
smaller head drafts faster; a token outside the subset is never drafted (the verify pass still emits it, so output
is unchanged, only speculation gets less help). --add-scripts takes more blocks for other languages.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.tokenizer import Tokenizer  # noqa: E402

BASE = [
    (0x0000, 0x007F),    # Basic Latin
    (0x0080, 0x00FF),    # Latin-1 Supplement
    (0x0100, 0x024F),    # Latin Extended-A/B
    (0x0250, 0x02FF),    # IPA, spacing modifiers
    (0x0300, 0x036F),    # combining diacritics
    (0x0370, 0x03FF),    # Greek (math, units)
    (0x1E00, 0x1EFF),    # Latin Extended Additional
    (0x2000, 0x206F),    # General Punctuation
    (0x2070, 0x209F),    # super/subscripts
    (0x20A0, 0x20CF),    # currency
    (0x2100, 0x218F),    # letterlike, number forms
    (0x2190, 0x23FF),    # arrows, math operators, misc technical
    (0x2460, 0x24FF),    # enclosed alphanumerics
    (0x2500, 0x25FF),    # box drawing, blocks, geometric shapes
    (0x2600, 0x27BF),    # misc symbols, dingbats
    (0x27C0, 0x2BFF),    # more math and arrows
    (0x2E00, 0x2E7F),    # supplemental punctuation
    (0xFE0F, 0xFE0F),    # emoji variation selector
    (0x1F300, 0x1FAFF),  # emoji
]
EXTRA = {
    "cyrillic": [(0x0400, 0x052F)],
    "han": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF)],
    "hangul": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    "cjk_punct": [(0x3000, 0x303F), (0xFF00, 0xFFEF)],
    "arabic": [(0x0600, 0x06FF)],
    "devanagari": [(0x0900, 0x097F)],
}


# lead bytes of multi-byte characters whose whole range lies in the kept blocks: a token may end part-way through
# such a character (English text splits em dashes and curly quotes, U+2014 = E2 80 94, across tokens)
def partial_ok(lead: bytes, ok_char) -> bool:
    n = {0xC0: 2, 0xE0: 3, 0xF0: 4}.get(lead[0] & 0xF0 if lead[0] >= 0xE0 else 0xC0, 0)
    if lead[0] < 0xC2 or n == 0:
        return False
    # every character the fragment can complete to must be kept: test both ends of its range
    lo = lead + bytes([0x80] * (n - len(lead)))
    hi = lead + bytes([0xBF] * (n - len(lead)))
    try:
        return all(ok_char(c) for c in (lo.decode("utf-8"), hi.decode("utf-8")))
    except UnicodeDecodeError:
        return False


def ok_bytes(b: bytes, ok_char) -> bool:
    """Whole characters all kept; a leading run of continuation bytes (the tail of a character begun in the
    previous token) is accepted; a trailing incomplete character must be one that can only complete to a kept one."""
    i = 0
    while i < len(b) and 0x80 <= b[i] < 0xC0:
        i += 1
    while i < len(b):
        c = b[i]
        n = 1 if c < 0x80 else 2 if c >> 5 == 6 else 3 if c >> 4 == 14 else 4 if c >> 3 == 30 else 0
        if n == 0:
            return False
        if i + n > len(b):
            return partial_ok(b[i:], ok_char)
        try:
            ch = b[i:i + n].decode("utf-8")
        except UnicodeDecodeError:
            return False
        if not ok_char(ch):
            return False
        i += n
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="first shard of the model (for its tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-scripts", default="", help="comma list of: " + ", ".join(EXTRA))
    args = ap.parse_args()
    ranges = list(BASE)
    for name in filter(None, args.add_scripts.split(",")):
        ranges += EXTRA[name.strip()]

    def ok_char(c: str) -> bool:
        o = ord(c)
        return any(a <= o <= b for a, b in ranges)

    tok = Tokenizer.from_gguf(args.gguf, cache_dir=str(Path("~/.cache/bnk").expanduser()))
    keep = []
    for i in range(len(tok.tokens)):
        if tok.types[i] not in (1,):  # anything but a normal token: control, user-defined, byte
            keep.append(i)
            continue
        b = tok._bytes_of(i)
        if len(b) == 1:
            keep.append(i)
            continue
        if ok_bytes(b, ok_char):
            keep.append(i)
    import array
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(Path(args.out).expanduser(), "wb") as f:
        array.array("i", keep).tofile(f)
    print(f"{len(keep)} of {len(tok.tokens)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
