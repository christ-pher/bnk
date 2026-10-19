#!/usr/bin/env python3
"""The toks path of serve/tokenizer.py against its Python path: same ids on a few corpora, and the time each takes.

    .venv/bin/python tools/tok_check.py --gguf model.gguf [--text FILE ...]

Each corpus is encoded by a fresh Python tokenizer (empty word caches: a first request's cost), by the same one
again (warm caches: a repeated text's cost), and by toks (a fresh context, then the kept one). Exits 1 on any
difference in ids or decoded text.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from serve.tokenizer import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--cache", default=os.path.expanduser("~/.cache/bnk"))
ap.add_argument("--text", action="append", default=[], help="a text file to encode (default: a few of the repo's)")
a = ap.parse_args()

files = a.text or [os.path.join(HERE, f) for f in ("README.md", "serve/server.py", "src/dense/dense_engine.cpp")]
corpora = {os.path.basename(f): open(f, encoding="utf-8").read() for f in files}
corpora["multilingual"] = ("Der schnelle braune Fuchs. Ça va très bien! ¿Dónde está? Привет, как дела? Σήμερα. שלום עולם. "
                           "مرحبا بالعالم. नमस्ते दुनिया. 今日はいい天気ですね。你好，世界！안녕하세요 세계! "
                           "Café naïve façade — 'quotes' “smart” … \t\n\n  code(x) != y; ") * 50
corpora["chat"] = ("<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n" + corpora["README.md"][:20000]
                   + "<|im_end|>\n<|im_start|>assistant\n<think>\n")

os.environ["BNK_TOKS"] = "0"
py = Tokenizer.from_gguf(a.gguf, cache_dir=a.cache)
os.environ["BNK_TOKS"] = "1"
tk = Tokenizer.from_gguf(a.gguf, cache_dir=a.cache)
print(f"python path: {py.backend}; toks path: {tk.backend}")
if tk.toks is None:
    sys.exit("toks is not available (pip install the wheel from github.com/actual-computer/toks/releases)")

bad = 0
print(f"{'corpus':22} {'bytes':>8} {'ids':>7} {'py cold ms':>11} {'py warm ms':>11} {'toks ms':>9} {'toks again':>11} {'speedup':>8}")
for name, text in corpora.items():
    os.environ["BNK_TOKS"] = "0"
    fresh = Tokenizer.from_gguf(a.gguf, cache_dir=a.cache)
    os.environ["BNK_TOKS"] = "1"
    t0 = time.perf_counter(); ids_py = fresh.encode(text); t1 = time.perf_counter()
    fresh.encode(text); t2 = time.perf_counter()
    fresh2 = Tokenizer.from_gguf(a.gguf, cache_dir=a.cache)
    t3 = time.perf_counter(); ids_tk = fresh2.encode(text); t4 = time.perf_counter()
    tk.encode(text); t5 = time.perf_counter()
    same = ids_py == ids_tk and py.decode(ids_py) == tk.decode(ids_tk)
    bad += not same
    print(f"{name:22} {len(text.encode()):8} {len(ids_py):7} {1e3 * (t1 - t0):11.2f} {1e3 * (t2 - t1):11.2f} "
          f"{1e3 * (t4 - t3):9.3f} {1e3 * (t5 - t4):11.3f} {(t1 - t0) / (t4 - t3):7.1f}x" + ("" if same else "   DIFFERENT"))
if bad:
    sys.exit(f"{bad} corpora differ between the paths")
print("same ids and text on every corpus")
