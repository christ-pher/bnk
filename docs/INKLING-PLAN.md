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
