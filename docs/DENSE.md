# The dense engine: Qwen3.8-27B on the two V100s (branch `dense`, 2026-10-18)

Goal: run the dense Qwen3.8 models (`unsloth/Qwen3.8-27B-GGUF`) in bnk without touching the Flash-Next (MoE)
code paths, at a quantization high enough to keep the model's quality, and as fast as the two V100s allow.

## What the model is

Qwen3.8-27B is not a classical dense transformer. Its GGUF architecture is `qwen35`, the Qwen3.5 hybrid that
Flash-Next also descends from:

| | Qwen3.8-27B | Qwen3.8-Flash-Next (bnk's MoE engine) |
|---|---|---|
| Layers | 64: 48 Gated DeltaNet + 16 gated full attention (every 4th) | 48: 36 + 12 |
| Width / FFN | 5,120 / 17,408, dense SwiGLU | 2,560 / 512 experts top-10 + shared |
| DeltaNet | 16 key groups x 48 value heads, head dim 128, conv 4 | the same shapes |
| Attention | 24 q heads / 4 KV heads, head dim 256, q|gate per head, RoPE on 64 dims | 24 / 2, QSA indexer |
| Residual | one stream, RMSNorm before each half, plain adds | 4 hyper-connection streams, PLE n-grams |
| MTP | one block after the trunk (block 64 of the GGUF, `nextn.*` tensors) | separate draft-layer GGUF |
| Vocabulary | 248,320 (tokenizer pre-type `qwen35`) | the same tokenizer family |

So the blocks bnk already had (DeltaNet conv / recurrence / gated norm, gated attention with the KV cache, the
quantized GEMVs, the fp16 tensor-core GEMMs of the prompt path, graphs, the drafter protocol) cover it; what is
new is the dense FFN, the plain residual, and a model with every weight in VRAM.

One difference that matters: the DeltaNet output gate of this family is **silu(z)** (`output_gate_type: swish`),
not sigmoid(z) as in Flash-Next. The first run produced garbage because of it; `gated_rmsnorm_silu` /
`quantize_gated_norm_silu` are the two added kernels. Two converter conventions were checked and found compatible:
the GGUF's norm weights carry the model's `1 + w` folded in (no `+1` at load, unlike `tools/build_mtp.py`'s
output), and the value heads are stored in llama.cpp's *tiled* order, which is exactly the `h % n_groups` mapping
bnk's recurrence assumes.

## Which quant

The two cards hold 64 GB. The GGUF sizes from the repo: Q8_0 27.05 GiB, UD-Q6_K_XL 23.6, UD-Q5_K_XL 19.4,
UD-Q4_K_XL 16.4, and BF16 51 GiB (too big beside a context). Q8_0 is the pick: near-lossless (8.5 bits per
weight), and it leaves ~17 GB per card for the context and the prompt path. The engine's layer outputs match
llama.cpp's fp32 CPU path within 0.3-1.2% RMS (that is the int8 activation quantization of the GEMVs; the weights
are used bit for bit), with the same top token at every position tested and a KL of 2e-4 on the logits.

## Where the code lives (and does not)

* `src/dense/`: `dense_model` (config, weight upload, tensor-parallel slicing), `dense_engine` (decode windows,
  graphs, slots, snapshots, pipeline or tensor parallel), `dense_prefill` (chunked prompt path), `dense_mtp` (the
  drafter), `dense_ops.cu` (the residual stream's fused norm + quantize, the adds).
* `src/engine/engine_api.h`: the interface the generator, server and command line talk to. The MoE `Engine` and
  `DenseEngine` both implement it; `bnk` picks one from the GGUF's `general.architecture`. That is the whole
  extent of the change to the shared code: `generate.cpp`, `serve.cpp` and `bnk_main.cpp` call the interface,
  `engine.h` declares the MoE engine as implementing it, and the two silu-gated kernels were added beside the
  sigmoid ones (not in their place). The Flash-Next regression (`Qwen3.8-Flash-Next-FP4-Local`, 3,000-token
  prompt, 300 greedy tokens with MTP) produces the same tokens on this branch as on master.
* `configs/Qwen3.8-27B-Q8_0.json` (aliases `qwen27b`, `27b`, `dense`): `"mtp": "builtin"` tells `run.sh` the
  drafter is inside the model file; `engine_args --tp 1` turns on tensor parallelism.

## How the two GPUs are used

**Pipeline (default of the engine, `--gpus 0,1`):** layers 0-33 on GPU 0, 34-63 on GPU 1 (balanced by bytes,
the head and the MTP block counted on the last card). Decode is bandwidth-bound: each card streams its 13 GB of
weights in ~16 ms, one after the other, 32.5 ms per forward. Prompts overlap in chunks (one stage works on chunk
k+1 while the other finishes chunk k).

**Tensor parallel (`--tp 1`, the config's setting):** every layer is split across both cards: half the attention
heads and KV heads, half the DeltaNet key groups with their value heads (the KV cache and recurrent state follow
their heads), half the FFN rows. Both cards stream their half at once; after the attention / DeltaNet and after
the FFN the two partial sums are all-reduced. Peer-to-peer copies between these two V100s take 51 ms whatever the
size (the PCIe topology under the hypervisor), so the exchange goes through pinned host memory: each rank copies
its partial out, waits for the peer's event, copies the peer's partial in and adds it. 23 us per exchange at one
row, 43 us at five; 128 of them per forward. Both ranks add the same two values, so their residual streams stay
identical bit for bit (prompt chunks exchange fp16 partials, and both ranks add the two rounded halves). One CUDA
graph per window layout spans both devices (rank 0 captures, the peer's stream joins through an event). Rank 0
holds the head and the drafter; the peer's residual is identical, so nothing needs to cross for them.

| Two V100s, Q8_0 | Pipeline | Tensor parallel |
|---|---|---|
| Decode, greedy, no drafting | 30.8 tok/s (32.5 ms per forward) | **42.4 tok/s** (23 ms) |
| Decode with the MTP drafter (draft 4 / min-p 0.8) | 80.6 tok/s, 3.4 tokens per round | **104 tok/s**, 3.4 per round |
| 3,000-token prompt | 981 tok/s (1,106 with the tensor-core attention below) | **1,193 tok/s** (**1,333**) |
| 300-token prompt | 507 tok/s | 825 tok/s |
| Served (`./run.sh qwen27b`, temperature 1.0, 545-token answer with thinking) | | 87.5 tok/s, 2.8 tokens per round |
| Decode at 64K / 250K of context (greedy, drafting) | | 84.8 / 53.9 tok/s |

The verify window of a speculative round carries up to 5 rows and takes 28 ms in TP (23 at one row): the GEMVs
are still weight-bandwidth-bound, the extra rows are nearly free, which is why drafting pays 2.5x here.

## The prompt path

cuBLAS fp16 GEMMs with each matrix dequantized to fp16 on the fly (as the MoE engine's dense projections), the
DeltaNet recurrence over the whole chunk, and a flash-attention kernel on the tensor cores
(`src/dense/dense_attn.cu`: mma.m8n8k4 with the GEMV's fragment layout; a block is 8 query rows x the 6 query
heads of one KV head, so a K/V tile staged in shared memory serves 6 warps; 32-key tiles, the online softmax
and the rescale of O in registers, V transposed in shared memory for the P V operands). Per block kind on a
3,000-token read (pipeline, GPU 0, 34 layers, five 600-token chunks): FFN ~1,430 ms (about 50 TFLOPS, near what
the 150 W power cap leaves of the tensor cores), DeltaNet ~680 ms, attention 214 ms, of which ~180 are the
q/k/v/o projections and ~35 the attention kernel (504 ms with the one-warp-per-row kernel it replaces;
`BNK_PF_ATTN_TC=0` brings that one back, and an 8-bit KV cache falls back to it). A 12,000-token read: 13.8 s
with the old kernel, 9.2 s with this one (1,309 tok/s, tensor parallel).

What Nsight Compute says about the kernel (2026-10-18): tensor pipe ~25% active, 12 resident warps per SM (168
registers and 38 KB of shared memory per 6-warp block), an instruction issued every ~4 cycles; the first version
also had 12-way bank conflicts on the transposed V stores (fixed: the staging threads run along the keys) and
masked every tile (now only the diagonal ones; exp2 with the scale folded in; the O rescale skipped when no row
max moved: together ~2.5x on the kernel). The next steps would be double-buffered K/V staging (the global loads
are the top stall) and 16 query rows per warp (halves the operand traffic per MAC), both of which need registers
the kernel does not have at 12 warps; at 3K-12K tokens the projections now cost more than the kernel, so this is
left as is.

## Long context (2026-10-18)

The user's conversations run to 256K tokens, so the engine was taken there. The hybrid helps: only 16 of the 64
layers keep a KV cache (64 KiB per token in fp16, 32 KiB in 8 bits), the DeltaNet layers' state is fixed-size,
so a 262,144-token context costs 16 GiB of KV in fp16 (8 per rank under tensor parallelism) or 8 GiB in 8 bits.
The cache is mapped as a conversation grows. The config now runs the model's native 262,144 with the 8-bit cache
and two slots, which fit beside the weights even with both conversations full; fp16 fits one.

Decode at depth was the first problem: the split-KV decode kernel of `kernels/ops.cu` reads the whole cache once
per window row and query head at a third of the memory bandwidth, so a 5-row verify window at 64K spent ~28 ms
per forward there. The tensor-core kernel above now serves decode too: one pass over each KV head's cache for the
window's rows and the 6 query heads of its group, the keys split over 64 blocks per head and combined (pos0 read
from device memory, so the forward graphs replay). The drafter uses it as well.

| Q8_0, tensor parallel, greedy with the MTP drafter | 64K context | 250K context |
|---|---|---|
| Cold read of the whole context | 67.5 s (970 tok/s) | 471 s (531 tok/s) |
| Decode at that depth, CUDA-core decode kernel | 56.6 tok/s | |
| Decode at that depth, tensor-core decode kernel | 84.8 tok/s | 53.9 tok/s |
| Verify window (5 rows) at that depth | 40 ms | 58 ms |
| Drafting per round at that depth | 7 ms | 13.5 ms |

What remains at 250K: the read is the attention kernel's 25% tensor-core utilization (the read's attention part
grows with the square of the length: ~5.5 of the 7.9 minutes), and decode is 54 instead of 104 tok/s because
every verify window and every draft step streams the 8 GiB of KV per rank (a verify window at 250K reads
~16 GiB across the two cards in ~30 ms: close to the bandwidth; the drafter's three attention passes per round
add ~10 ms). A follow-up turn reads only its new tokens: the engine snapshots the recurrent state where the last
turn began and resumes from there, as the MoE engine does. The engine has no parking: a third conversation
arriving while two slots are full is read from scratch, and the slots are not moved to host RAM.

## Serving

`./run.sh qwen27b` (the config's `server_args`: context 131,072, 2 slots, no parking: parking exists for the MoE
engine's expert cache pressure and is a no-op here; the KV cache is mapped on demand like the MoE engine's). The
server's telemetry fields about experts read 0 and the console shows no "CPU misses" line, since nothing runs on
the CPU.

## Open items

1. The tensor-core attention kernel's efficiency (above) for 32K+ prompts.
2. fp16 exchange in decode windows too (halves the bytes of the 5-row verify windows: 43 -> ~30 us per
   exchange, ~1.5 ms per forward).
3. The draft head: the drafter reads the 1.3 GB Q8_0 head (over the draft vocabulary's 145K rows re-quantized to
   Q4_K, as the MoE engine does) once per draft step; a smaller draft vocabulary or a lower quant of the draft
   head would cut the ~1 ms per step.
4. Sliced head: the head's rows could be split between the ranks too (0.8 ms per forward on rank 0 today).
5. Lower quants (UD-Q6_K_XL, UD-Q5_K_XL): the loader is format-agnostic (any ggml type the R layouts support);
   a K-quant's 256-wide blocks fit the 1,024-wide slices tensor parallelism makes. Untested.
