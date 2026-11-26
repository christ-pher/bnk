# Inkling-2-Mini port (branch `inkling`)

Thinking Machines' Inkling 2 Mini (2026-11-20, Apache 2.0): a multimodal looped MoE transformer, 26.2B parameters,
3.4B active a layer visit (5.4B effective with the loop). GGUF `unsloth/Inkling-2-Mini-GGUF`, the largest quant
UD-Q8_K_XL (21.3 GiB: experts NVFP4 with per-expert fp32 scales, everything else BF16 / F32, the last layer's
experts BF16). Architecture `inkling2_loop`; reference: mainline llama.cpp `src/models/inkling2.cpp` and
`inkling2-loop.cpp` (PR #32694, the per-pass NVFP4 scale fix #32704), the relative-position attention in
`llama-graph.cpp` (`build_inp_mem_hybrid_iswa_rel`, `build_attn_mha` with `bias_table`).

## The model

* 24 physical layers, E = 2304, 24 query heads / 8 KV heads of 128, 128 experts (8 used, FFN 1152, gate | up
  merged), vocabulary 200,058 (tables padded to 201,024), no RoPE.
* **The loop**: layers 0-5 (prelude), 6-17 (core) three times with shared weights, 18-23 (coda): 48 layer visits a
  token. Every visit has its own KV cache and convolution state ("cache slots": 24 + 2 x 12 = 48). Passes 2 and 3
  start again from the prelude output; each core layer then mixes a *message* (a depth mix over the previous pass's
  bank: its block after layer 11 and its output, with the layer's 4 `loop_msg_query` rows) into its Q, K, V and FFN
  inputs: `cur - tanh(forget(g)) * cur + tanh(gain(g)) * msg`, g = down([rms(cur); rms(msg)]) (rank 64).
* **Block attention residuals** (block size 6): at visits whose layer index % 6 == 0 the layer input is pushed to
  a list of blocks and the attention output starts a new stream (no residual add). Before the attention (layers > 0:
  1 query, full-attention layers 3, one each for Q, K, V) and before the FFN (1 query) the input is a softmax mix
  over [rms(blocks...), rms(cur)] by a learned query (scale 1/sqrt(E)). The output head mixes the same way.
* **Attention**: 5 of 6 layers a 1,024-token window (one input norm), every 6th (5, 11, 17, 23) full (separate
  Q / K / V input norms, input-dependent conv taps). Q: per-channel scale, then L2 clipped to 22.63 (sqrt 512);
  K: L2 clipped; V: RMS-normed per head. K and V (and the attention output, the FFN output) pass a causal short
  convolution of width 4 with a residual, scaled 1/sqrt(2*4+1) (dynamic) or 1/sqrt(4+1). Scores q.k / sqrt(128)
  plus a bias b(d) = R . rel_bias[:, min(d, extent)] - sink, R = clip(attn_rel(x)) (16 per head, clipped at 8),
  rel_bias zero-padded at d >= extent (window: 1,024; full: 1,024), sink = attn_sink_proj(x) per head, and a zero
  logit in the softmax's denominator. Full layers scale q and R by 1 + 0.1 ln(max(1, (pos+1)/8192)). Output gated
  by sigmoid(attn_gate(x)).
* **MoE**: router F32, sigmoid; top-8 by sigmoid + exp_probs_b; weights the sigmoids normalized (sum + 1e-20) x 8.
  SITU: 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25).
* Embedding RMS-normed (tok_norm); the head's logits soft-capped at 30.
* Tokenizer: GPT-4o pre-split (`inkling2` = llama.cpp's GPT4O, ignore_merges), no BOS. Template: `<|message_*|>`
  roles, `<|content_text|>`, `<|content_thinking|>`, tool calls `<|message_model|>NAME<|content_invoke_tool_json|>
  {"name":..,"args":..}<|end_message|>`, end of turn `<|content_model_end_sampling|>`; thinking effort a system
  line (0..0.99).
* Drafter: a 3-layer DSpark block drafter (block 8) that taps the looped layers (later).
* Vision: mmproj (`inkling2v` in llama.cpp's mtmd; later).

## Steps

1. Download (done): `Inkling-2-Mini-UD-Q8_K_XL.gguf`, `mmproj-F16.gguf`, `dspark-Inkling-2-Mini-Q8_0.gguf`.
   A CPU build of mainline llama.cpp as the reference (logits and intermediate tensors on a short prompt).
2. `src/inkling`: the model loader (every weight on one GPU: ~21 GiB of a V100's 32), the forward over a window of
   tokens and prompt chunks, single slot; new kernels (depth mix, short conv with state, clip, relative-bias
   attention with sinks over a window ring and a full cache, SITU MoE, loop control, soft cap). Check against the
   reference's logits.
3. Server: the tokenizer (GPT-4o split, ignore_merges), the template's markers (thinking, tool calls, end of turn),
   a config. Then decode speed: CUDA graphs, the MoE kernels for resident NVFP4 experts, two GPUs.
4. Later: the DSpark drafter, vision, conversation slots / parking.

## Log
* 2026-11-25: steps 1-3 (first part) done on one V100.
  * **Correctness** (tests/test_inkling against llama.cpp's CPU build, `ref_logits`): 26-token prompt top-1 100%,
    mean KL 0.0003; 2,934-token markdown/code prompt top-1 96.8%, mean KL 0.005, flat over positions (no step at the
    512-token chunks, the 1,024-token window or the relative-bias extent), next-token NLL 2.1124 vs the reference's
    2.1101. Decode, prompt chunks and part-committed verify windows (8 rows, 1-8 kept) give the same logits across
    the window boundary. The reference rounds BF16 products' activations to BF16 (BNK_INK_BF16ACT=1 does the same
    here: KL 0.0043 -> 0.0040, so not the gap); the rest is the two implementations' fp32 / fp16 / int8 noise.
  * **Server**: tokenizer exact (3,125 tokens of mixed text), template, InklingParser (thinking, text, JSON tool
    calls), thinking levels, the loop guard's wrap-up, snapshots at the prompt's end (a tool turn reused 213 of
    278 tokens). The generator now verifies prompt-lookup copies without a drafter (BNK_LOOKUP_ALONE=0 off).
  * **Speed** (decode, one V100, the 150 W cap): 36 tok/s first (host round trip per layer visit for the MoE) ->
    47.6 tok/s prose: the MoE on the device (a plan grouping the window's pairs by expert, gate | up and down as
    dp4a products on int8 activations, NVFP4 rows read once per expert: 450 GB/s; fp32 activations were shared-
    memory bound at ~300 GB/s), positions in device memory and one CUDA graph per window size, the depth mix in
    two passes. Copy tasks 88 tok/s (lookup drafts: 769 of 828 accepted, 2.98 tokens a round). Prompts: ~330 tok/s
    on short reads (the grouped GEMM path; not tuned yet).
  * **Where decode time goes** (nsys, 19.8 ms a token): BF16 projections 4.7 ms (~560 GB/s), NVFP4 experts 3.9 ms,
    ~2,000 small kernels a token (norms, mix scores, clipping, convolutions, loop controls: each latency bound at
    3-10 us), attention 0.9 ms, layer 23's BF16 experts 0.6 ms (shared-memory bound). The weight traffic is ~5.5 GB
    a token: ~110-120 tok/s is one V100's ceiling.
  * **Next**: fuse the small kernels (mix + norm, the three clips, k | v conv, the q / k / v loop controls),
    split-K attention, prompt speed (the MoE prompt path: moe_tc-style grouped tensor-core experts, flash prompt
    attention), the DSpark drafter (dspark-Inkling-2-Mini-Q8_0.gguf, block 8, taps the looped layers), vision
    (mmproj-F16.gguf), the second GPU (tensor parallel would halve the weight traffic), more slots.

## Overhaul (2026-11-25, the user: "only one of the GPUs was being used")

1. Attention: decode split over the keys (a block per KV head, window row and key range; the three query heads of
   a KV head together, K/V tiles in shared memory), a combine pass; prompts: tensor-core flash attention with the
   relative bias, the sink and the window. Long contexts were the cliff (40K: decode 15.8 tok/s, prompts 305 tok/s).
2. Tensor parallel over both V100s: half the query / KV heads (12 / 4), half of every expert's FFN width (576 of
   1,152), half the vocabulary each; the depth mixes, norms, loop controls and routers on both. Two all-reduces a
   visit (after the attention's output projection, after the MoE) of [T][E] over P2P, captured in one graph.
3. Prompt-path MoE on tensor cores (grouped by expert), bigger chunks.
4. Fewer, fused small kernels.
5. The DSpark drafter.

### Overhaul results (2026-11-25, through the server, both GPUs)

| prompt | read before -> now | decode before -> now |
|---|---|---|
| short | - | 47.6 -> 55 tok/s |
| 2.9K | 507 -> 1,046 tok/s | 37.4 -> 52.8 tok/s |
| 11K | 491 -> 1,100 tok/s | 28.8 -> 51.2 tok/s |
| 40K | 305 -> 1,040 tok/s | 15.8 -> 48.1 tok/s |

Context 131K -> 262K (each GPU holds half the KV cache: 4.1 GiB at 262K, 15.5 GiB free). Logits still match the
reference (2,934 tokens: NLL 2.1130 vs 2.1101, top-1 97.0%).

* Steps done: 1 (tensor-core attention: the long-context cliff), 2 (tensor parallel: half the weights a GPU), 3
  (tensor-core experts in the prompt path), part of 4 (stacked projections, fused routing + plan, mix + norm in one
  block: 2,426 -> 1,977 launches a token).
* Decode is now bound by ~2,000 dependent small kernels a token on each GPU (14.4 ms of work at 82% busy; the
  weights themselves are ~4 ms a GPU): tensor parallel halved the weight reads but the replicated small kernels and
  the 96 exchanges stay. The biggest lever left is verifying several tokens a window (the small kernels cost the
  same for 4 rows as for 1): the DSpark drafter (step 5).

### Merged kernels and the drafter (2026-11-25, later)

* Merged kernels (1,977 -> 1,274 launches a token on each GPU): convolutions as jobs of one launch (the exchange's
  add folded into their input, the residual / copies into their output), the clips and v's norm in one, the loop
  controls batched with their norm folded into the combine, the decode mix holding every state in registers. Plain
  decode (two GPUs) 57.4 -> 68.6 tok/s.
* The DSpark drafter (src/inkling/ink_spark): checked against a NumPy reference of llama.cpp's graphs (identical drafts
  from the same features) and its taps against llama.cpp's own tensors (l_out-4 / -28 / -46, result_norm); greedy
  acceptance as llama.cpp's on the same prompts (prose ~0.8, code ~1.6 drafts a round). The first try accepted 5%:
  the three stream taps were never captured (a no-op edit), only the normed output.
* Each verify row costs ~2.6 ms (its 8 experts' weights a visit), so 2 drafts a round is fastest; the drafter stops
  its block at the round's drafts (Drafter::set_max_drafts): 3.3 ms a round.
* Through the server (temperature 1): prose 68, code 92, explanations 86 tok/s; after an 11K prompt 80, after 40K 69
  (55 / 51 / 48 before the drafter).
* Next: the drafter's head and Markov bias split over both GPUs (GPU 1 idles while GPU 0 drafts), its block as a
  graph, cheaper verify rows (the window's experts), vision, more slots.

### Next steps, first pass (2026-11-26)

* Top-k over long rows in two stages (shared kernel: the sampler's head and the drafter's draws): K 64 460 -> 81 us,
  K 1,024 472 -> 217 us for 3 rows of 200K.
* The loop-control down product a block per row, the bias table a block per row for decode windows: plain decode
  68.6 -> 71.9 tok/s.
* The drafter's head over both GPUs (each rank's own vocabulary rows: no copy of the head, -0.9 GiB), its draft as one
  graph per block size: 3.3 -> 1.7 ms a round. Drafts identical to the NumPy reference.
* Through the server (temperature 1): prose 68 -> 78, code 92 -> 101, explanations 86 -> 100 tok/s; after 11K 80,
  after 40K 69 -> 77.
* Where a round goes now (2 drafts): verify ~19 ms (windows of 1 / 2 / 3 / 4 rows: 14.0 / 15.8 / 17.4 / 18.7 ms with
  the same experts; different tokens add their experts), draft 1.7 ms. The BF16 GEMV re-reads x from L1 per token
  (L1-bound for T > 1: +0.75 ms a window from 1 to 3 rows); the rest of the per-row cost is spread thin.
* Not yet: images (inkling2v: a ViT with a CLS token, resized learned position embeddings, 2D RoPE, patch merge,
  GELU projector; image rows enter as embeddings without the token norm), more conversation slots.
