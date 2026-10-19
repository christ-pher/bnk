"""A Hugging Face tokenizer.json built from a GGUF's tokenizer metadata, for the toks library (serve/tokenizer.py).

The GGUF carries the vocabulary, the merges and the token types of the model's tokenizer.json, but not its
pipeline (normalizer, pre-tokenizer, decoder): llama.cpp names the pipeline by `tokenizer.ggml.pre` instead.
This module writes the pipeline back for the families it knows, so the file toks loads says exactly what the
model's own tokenizer.json said. Checked for qwen35 against Qwen/Qwen3.8-27B's file (byte-identical to the
Flash-Next checkpoint's): same vocabulary, same merges, same ids on every corpus tried.
"""
from __future__ import annotations

import json

from .gguf_meta import read_metadata

# llama.cpp's pre-tokenizer names -> the tokenizer.json pipeline (normalizer, Split regex)
QWEN35_RE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
             r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
PIPELINES = {
    "qwen35": {"normalizer": {"type": "NFC"}, "regex": QWEN35_RE},
}

# llama.cpp token types
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6


def build_tokenizer_json(md: dict) -> dict | None:
    """The tokenizer.json dict for a GGUF's metadata (tokens, merges, token_type, pre), or None when the
    pre-tokenizer family is not one PIPELINES knows (the Python tokenizer then runs as before)."""
    pipe = PIPELINES.get(md.get("tokenizer.ggml.pre", ""))
    if pipe is None:
        return None
    tokens = md["tokenizer.ggml.tokens"]
    types = md.get("tokenizer.ggml.token_type") or [NORMAL] * len(tokens)
    vocab, added = {}, []
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if ty in (CONTROL, USER_DEFINED):
            added.append({"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": ty == CONTROL})
        elif ty in (NORMAL, BYTE):
            vocab[t] = i
        # UNUSED ([PAD...] rows that pad the embedding table) and UNKNOWN are no tokenizer's ids
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": pipe["normalizer"],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pipe["regex"]}, "behavior": "Isolated", "invert": False},
            byte_level]},
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": list(md["tokenizer.ggml.merges"])},
    }


def write_tokenizer_json(gguf_path: str, out_path: str) -> bool:
    """Writes the GGUF's tokenizer as tokenizer.json; False (nothing written) for an unknown family."""
    md = read_metadata(gguf_path, {"tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
                                   "tokenizer.ggml.pre"})
    j = build_tokenizer_json(md)
    if j is None:
        return False
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, separators=(",", ":"))
    return True
