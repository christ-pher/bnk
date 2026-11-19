# LoRA adapters on the dense engine (`--lora`, 2026-11-18)

The dense engine (Qwen3.8-27B and the rest of the `qwen35` family) applies a LoRA adapter at run time: the base GGUF
stays as it is and the adapter is a separate ~300 MB file. The first one is
[harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted): rank 32, alpha 64,
on q/k/v/o of the 16 attention layers and gate/up/down of all 64 FFNs (not the DeltaNet projections, the MTP block or
the vision tower). It was trained with RL as the model behind the [pi](https://github.com/earendil-works/pi) coding agent:
SWE-bench Pro (hard 51) 39.2% -> 64.7% and Terminal-Bench 2.1 39.5% -> 55.8% under pi, with thinking off.

## Running it

```sh
mkdir -p ~/.cache/bnk/pi-adapted
.venv/bin/hf download harness-hub/Qwen3.8-27B-pi-adapted-GGUF Qwen3.8-27B-pi-lora-F16.gguf --local-dir ~/.cache/bnk/pi-adapted
./run.sh pi            # configs/Qwen3.8-27B-pi-Q8_0.json: the Q8_0 27B (the same file as qwen27b) + the adapter
```

The config's `"lora"` is a file in the cache directory (or a path); `"lora_scale"` multiplies the adapter (0 gives
the base model back, bit for bit). Sampling is the training's (temperature 0.8, top-p 1.0) and thinking is off by
default (`"chat_template_kwargs": {"enable_thinking": false}`, the equivalent of vLLM's
`--default-chat-template-kwargs`). A request still turns thinking on with its own `chat_template_kwargs`, a top-level
`enable_thinking` or a `reasoning_effort` other than `none` (an explicit `enable_thinking` beats `reasoning_effort`;
tests/test_thinking_default.py). The model card says thinking adds nothing to the adapter's accuracy and costs about
4x the tokens. A small check of our own agrees (2026-11-19, below: thinking off 19/20, on 18/20 at 3.3x the output).

pi (`~/.pi/agent/models.json`; `"api": "openai-completions"` matters, the adapter was not trained on the Responses API,
and `"reasoning": false` keeps pi from asking for thinking):

```json
{ "providers": { "bnk": { "baseUrl": "http://localhost:8080/v1", "api": "openai-completions", "apiKey": "1",
  "models": [ { "id": "Qwen3.8-27B-pi", "contextWindow": 262144, "maxTokens": 32768, "reasoning": false } ] } } }
```

`pi --provider bnk --model Qwen3.8-27B-pi`

`"reasoning": false` is also pi's default for a models.json entry: pi then sends no thinking parameters and the
config's default (off) applies. With `"reasoning": true` pi offers its thinking levels and sends one as
`reasoning_effort`, which turns thinking on here; its level `off` sends nothing.

## Why at run time, not merged

Merging the adapter into the Q8_0 weights does not work. The delta is 0.1-0.2% of each weight, under half a Q8_0 step
for >99.7% of them, and dequantized Q8_0 weights sit exactly on the grid, so re-rounding W_q8 + dW returns W_q8 almost
everywhere: the delta that survives points cos 0.07-0.23 along the adapter's (measured on blk.3 ffn_down, blk.40
ffn_up, blk.3 attn_q). A correct merge starts from the BF16 weights (54.6 GB to download) and writes a new 27 GB
Q8_0; the run-time adapter needs neither and applies the delta in full precision.

## How it is applied

`DenseModel::load_lora` reads the adapter GGUF (llama.cpp's `convert_lora_to_gguf.py` format: `<tensor>.lora_a`
[r][in], `<tensor>.lora_b` [out][r], `adapter.lora.alpha`). Per layer there are up to four sites, each a group of
projections reading one input: q | k | v, o, gate | up, down. A site's A matrices are stacked (Q8_0); its B matrices
are F16 with alpha / r folded in. Tensor parallel: row-split projections (q, k, v, gate, up) keep the whole A and their
rows of B; column-split ones (o, down) keep their columns of A and the whole B, so B (A_r x_r) lands in the partial
sum before the all-reduce. A tensor the engine does not adapt (a DeltaNet projection, say) is an error, not skipped.

Decode: when the base matrix is Q8_0, A's rows are stacked under it at load (q | A, o | A, gate | up | A, down | A,
rebuilt from the GGUF's bytes; the base QMats become row views), so one GEMV computes the base product and A x
together, the rows from the base's end on going to a third `GemvSplit` output. One kernel (`lora_up`, dense_ops.cu)
then adds the site's B products into their outputs. Those GEMVs run on the tensor cores at every window size
(`GemvSplit::mma_min`): through the dp4a kernels' int8 activations A x is off by 0.4-0.8% (tests/test_lora.cu) and
the adapter's effect blurs (below). The prompt path: two cuBLAS products per site beside the base GEMM.

Switches: `BNK_LORA_STACK=0` (A as its own GEMV on a side stream), `BNK_LORA_FUSED=0` (one float GEMV per B, the
reference for `lora_up`), `BNK_LORA_MMA_MIN=N` (the tensor cores from N rows; 2 saves ~1 ms per one-row forward and
gives up the accuracy there).

## Checks (2026-11-18, two V100s, tensor parallel)

Against llama.cpp's CPU path with the same adapter (`tools/llama_ref.cpp`, `REF_LORA=...`; `bnk check`,
`BNK_CHECK_T` for the window), a 60-token chat prompt:

| | layer 15 rel RMS | layer 63 rel RMS | KL(ref ‖ bnk) mean / max | argmax |
|---|---|---|---|---|
| base engine vs llama.cpp base | 5.8e-3 | 0.114 | 0.0019 / 0.025 | 59/60 |
| adapter vs llama.cpp + adapter | 6.0e-3 | 0.104 | 0.0021 / 0.034 | 57/60 |
| adapter vs llama.cpp base (control) | | 0.275 | 0.0110 / 0.222 | 56/60 |
| adapter, one-row windows, dp4a A x (before `mma_min`) | | 0.150 | 0.0039 / 0.110 | 58/60 |
| adapter, one-row windows, tensor cores | | | 0.0021 / 0.034 | 57/60 |

`--lora-scale 0` reproduces the base model's NLL exactly (0.58229 on 500 tokens of source after 1,500 of prompt).
The prompt path agrees with the decode path (NLL 0.58473 both, prompt read in chunks or through decode windows).
`tests/test_lora.cu`: `lora_up` within 3e-8 of fp64, the stacked GEMV identical bit for bit to the separate ones.

Speed, one forward (ms; `bnk nll`, Q8_0, TP):

| window rows | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| base | 21.8 | 23.3 | 26.9 | 26.2 | 28.9 |
| adapter | 24.0 | 25.2 | 26.8 | 27.9 | 31.0 |

So +7-10% on the windows drafting runs (1-2 and 5-8 rows), nothing at 3 (the tensor-core GEMV beats dp4a there; the
base's default threshold is 4). How the cost came down: A as its own GEMV and one float GEMV per B (416 extra launches)
+3.0 ms per 5-row forward; A on a side stream bought nothing (the base GEMV holds every SM); A stacked under the
base matrix and the B products in one kernel (`lora_up`, rank and window rows compiled in) +2.0 ms. What is left is
mostly `lora_up` (~7 us x 160 sites; latency, not bandwidth); folding the B products into the kernels that read
their outputs next (`quantize_act_silu_mul`, the residual add) would remove it.

The first pi run (a size parser that stopped at the decimal point): found, fixed with a one-line change and the
tests run, in 6 s and four requests.

Thinking off vs on under pi (2026-11-19): ten small Python repos (eight bugs, two features from a docstring), each
run twice per mode with `pi -p`; pi saw some tests, the grade was those (restored, in case they were edited) plus
hidden ones. Thinking on is pi's level `high` (`"reasoning": true`).

| | passed (visible + hidden) | output tokens, median / total | seconds, median / total | requests |
|---|---|---|---|---|
| thinking off | 19/20 (visible 20/20) | 563 / 13,389 | 8.0 / 181 | 94 |
| thinking on | 18/20 (visible 20/20) | 1,866 / 43,635 | 20.1 / 486 | 123 |

The misses: thinking on both times left the rate limiter's second bug (a refused call did not advance the clock), which
thinking off fixed both times; thinking off once wrote a Roman parser that rejected IV. No run edited the tests. Too
small to rank the two (the difference is one run), but it matches the model card: thinking buys nothing here and costs
3.3x the tokens and 2.7x the time.
