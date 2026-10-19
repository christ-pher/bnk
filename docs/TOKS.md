# The toks tokenizer in the server

[actual-computer/toks](https://github.com/actual-computer/toks) is a C tokenizer library (Apache-2.0) with AVX2 and
NEON kernels that returns exactly the ids Hugging Face `tokenizers` returns for a model's `tokenizer.json`. bnk's
server tokenized in Python until now (`serve/tokenizer.py`: the `regex` pre-tokenizer and a Python BPE merge loop
with word caches, plus `encode_chat`, which reuses a recent prompt's ids up to the last special token they share so
a follow-up turn encodes only its new text). Since 2026-10-19 the same module encodes through toks when its wheel
is installed, and falls back to the Python path otherwise. The ids are the same either way.

## What changed

- **`serve/tokenizer_json.py`** writes the GGUF's tokenizer as a `tokenizer.json`: the vocabulary, merges and token
  types come from the GGUF, the pipeline (NFC normalizer, the Split regex, ByteLevel pre-tokenizer / decoder /
  post-processor) from a table keyed by llama.cpp's `tokenizer.ggml.pre` (`qwen35` today). For the two Qwen 3.8
  models the result loads in toks with the same vocabulary, merges and ids as `Qwen/Qwen3.8-27B`'s own file (which
  is byte-identical to the Flash-Next checkpoint's). The file is written once beside the tokenizer pickle in
  `~/.cache/bnk` (`tok-<key>.json`, 9 MB). `tools/gguf_tokenizer_json.py` writes it by hand.
- **`serve/tokenizer.py`** loads toks over that file at `from_gguf` (`_attach_toks`: a vocabulary spot check, then
  `toks.Tokenizer.from_file`). `_encode_plain` (the text between added tokens) and `decode` go through toks; the
  added-token split, `encode_chat`'s prefix reuse and the stream decoder are unchanged. `BNK_TOKS=0` keeps the
  Python path. The server's ready line names the backend (`tokenizer toks 0.3.2 (avx2)` or `python`), and each
  response's `timings` carries `tokenize_ms`, the time the request's prompt took to encode.
- **`setup.sh`** installs the wheel for the venv's CPython from the toks GitHub release (version pinned, sha256
  checked against the release's `SHA256SUMS`), and goes on without it when there is none for the platform
  (`BNK_NO_TOKS=1` skips the step). The wheels are not on PyPI, hence not in `requirements.txt`.
- **`tools/tok_check.py`** encodes a few corpora through both paths and reports ids equal (or not) and the times.

Why the GGUF route rather than downloading the model's `tokenizer.json`: the GGUF is the one file every bnk model
has, its vocabulary is what the model's head was built against, and a quantizer's repo does not always ship the
tokenizer (unsloth's GGUF repos don't). The pipeline table has one entry; a GGUF of another family loads with the
Python path, which has the same limitation (its regex is `qwen35`'s).

## Tokenizer alone

`tools/tok_check.py` on the EPYC 7402 (Zen 2, AVX2, one thread), ids identical on every corpus:

| corpus | bytes | ids | Python, first request | Python, caches warm | toks | speedup (first request) |
|---|---:|---:|---:|---:|---:|---:|
| README.md (prose) | 44,697 | 13,320 | 34.8 ms | 7.9 ms | 1.33 ms | 26x |
| serve/server.py (code) | 41,393 | 10,412 | 28.5 ms | 7.1 ms | 1.24 ms | 23x |
| 65K-token prompt | 208,729 | 65,472 | 85.2 ms | 39.7 ms | 4.6 ms | 18x |
| 250K-token prompt | 802,754 | 250,000 | 205 ms | 152 ms | 10.0 ms | 21x |
| multilingual + CJK | 16,800 | 3,851 | 5.2 ms | 1.9 ms | 1.4 ms (0.17 ms kept) | 3.6x |
| chat with special tokens | 20,554 | 6,277 | 22.5 ms | 3.8 ms | 0.69 ms | 33x |

Decoding 250K ids: 13 ms in Python, 1.3 ms in toks. toks's own machines are faster (its README: 270-470 MB/s on a
Zen 5 or Cortex-X925 core against 30-95 MB/s here, from the slower core and Python's call overhead on short texts).

## End to end

The tokenizer's share of a turn was small before: a 256K-token first turn encoded in 0.2-0.3 s in Python against a
read of 123 s (Flash-Next) or 512 s (the 27B), and a follow-up turn encodes only its new text. The LocalMaxxing
sweep (`/opt/benchmarks/localmaxxing_bench.py`: a cold read, then a follow-up turn, at each depth, one pass, greedy,
512 tokens decoded) on both models from this branch's server, the tokenizer in toks and in Python:

| model, depth | turn | tokenize, Python -> toks | time to first token, Python -> toks |
|---|---|---|---|
| FP4-Local 4K | cold / follow-up | 11.8 -> 14.3 ms (first request, toks warming up) / 1.7 -> 0.3 ms | 2.99 -> 2.97 s / 1.00 -> 0.98 s |
| FP4-Local 64K | cold / follow-up | 109 -> 6.2 ms / 1.8 -> 1.0 ms | 26.6 -> 26.7 s / 1.12 -> 1.11 s |
| FP4-Local 256K | cold / follow-up | 293 -> 19 ms / 5.2 -> 4.0 ms | 126.1 -> 123.6 s / 1.50 -> 1.55 s |
| Qwen3.8-27B 4K | cold / follow-up | 8.9 -> 0.7 ms / 1.7 -> 0.2 ms | 3.22 -> 3.19 s / 0.60 -> 0.59 s |
| Qwen3.8-27B 64K | cold / follow-up | 110 -> 6.1 ms / 1.8 -> 1.2 ms | 69.5 -> 68.7 s / 0.93 -> 0.92 s |
| Qwen3.8-27B 256K | cold / follow-up | 269 -> 20 ms / 4.3 -> 4.2 ms | 513.5 -> 512.1 s / 2.04 -> 2.04 s |

So: 15-20x less time in the tokenizer on cold prompts, which is 0.1-0.3 s off a 64K-256K first turn and
nothing measurable on follow-ups (their encoding was already a few milliseconds through `encode_chat`'s prefix
reuse). Decode speeds are unchanged (the differences in the run are single-pass noise of the expert cache's
state after a cold read). The change is worth having for its exactness guarantee and for what it removes
(a Python BPE loop and its caches from the request path), not for the turn times.
