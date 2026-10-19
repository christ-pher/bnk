#!/usr/bin/env python3
"""Writes a model's tokenizer.json from its GGUF (serve/tokenizer_json.py), the file the toks library loads.

    .venv/bin/python tools/gguf_tokenizer_json.py --gguf model.gguf --out tokenizer.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve.tokenizer_json import write_tokenizer_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if not write_tokenizer_json(a.gguf, a.out):
    sys.exit("this GGUF's pre-tokenizer family is not one serve/tokenizer_json.py knows")
print(a.out)
