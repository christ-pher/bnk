# Cold experts in smaller formats (2026-10-19)

Item 1 of the model-side ideas (`docs/PERF-IDEAS-2026-10-19.md`, continued in the session notes): FP4-Local keeps
75-79% of its experts in VRAM; the quarter outside costs a PCIe copy of every one of them per prompt chunk (the
bound on its reads, 0.75 s per short turn) and CPU time on every decode miss. Routing is skewed: on this
machine's traffic the coldest 30% of each layer's experts carry 4.7% of routings and the next 40% carry 22%
(`tools/expert_bands.py` on the routing-count file). Storing those bands in smaller formats makes every expert
fit, at a quality cost paid only by the rare routings into them.

## What there is

- **`tools/expert_bands.py`**: the routing share of the coldest bands, from `~/.cache/bnk/counts-<model>.bnkc`
  (the file the server accumulates over real traffic and saves every minute).
- **`build/requant_experts`** (`tools/requant_experts.cpp`): writes the cold bands as a GGUF beside the model.
  `--bands iq2_s/q2_0:30,iq3_s/q2_0:40` means: per layer, the coldest 30% of experts with gate/up in IQ2_S and
  down in Q2_0, the next 40% with gate/up in IQ3_S and down in Q2_0. Each expert is dequantized from the model's
  own blocks (NVFP4's per-expert global scale folded in) and re-quantized with ggml's quantizers; the down
  matrices have 640 columns, which the 256-wide IQ and K formats cannot hold, so they take a 32/64-wide format
  (Q2_0 at 2.25 bits, or IQ4_NL / Q4_0 / Q8_0 for a lossless-but-no-saving band). Output tensors:
  `blk.N.cold_<band>.experts` (the ids) and `blk.N.ffn_{gate,up,down}_exps.<band>`. 24 threads, about 15 minutes
  for FP4-Local.
- **The engine** (`--cold-experts FILE`; config key `cold_experts`, a file name in the cache directory;
  `BNK_COLD_EXPERTS=0` serves without it): `Model::load_cold` binds each layer's bands; `ColdStore` holds their
  blobs in pinned RAM; `ExpertCache` gives each band a region of its own, sized for its format, filled first and
  never shrunk, and numbers its slots after the hot ones (`slot_of` ≥ `n_expert`); the hot ranking and the
  adaptive swaps run over the other experts and whatever VRAM is left. The MoE kernels take a slot range per
  region (`MoeLayerDesc::slot_lo / slot_hi`): a layer's decode step runs gate/up for the hot region and each
  band, the activation once (cold entries skip the NVFP4 global scale, which their blocks carry), then down per
  region; the prompt path builds its tensor-core items per region the same way. CPU misses can only be hot
  experts, so the host expert pool is unchanged.

## Keeping up with the routing

The file fixes which experts are cold. `run.sh` keeps it current without anyone watching: at every start,
`tools/cold_drift.py` compares the file's bands with the bands today's counts would choose (the share of cold
slots whose expert changed); over 10%, or when the file is missing, `build/requant_experts` runs in the
background on 12 threads (the live server keeps its mapped copy; the new file replaces the old one atomically
and the next start reads it). The config's `cold_bands` sets the bands (default `iq2_s/q2_0:30,iq3_s/q2_0:40`);
`BNK_COLD_AUTO=0` leaves the file alone. Day-to-day count changes do not cross band boundaries, so in practice the
rewrite happens after a real change of workload.

## Bytes

NVFP4 experts are 4.5 bits per weight (2.8 MB each, 63 GiB in all). Per demoted expert the mix saves 46% (IQ2_S
gate/up + Q2_0 down) or 32% (IQ3_S gate/up + Q2_0 down); the 30/40 bands save 27% of the expert bytes, against
the 21-25% that lived outside VRAM. The cold file is about 30 GB on disk and the same in pinned RAM beside the
NVFP4 store (the machine has 192 GB).

## Measured

Quality is `bnk nll` over 10,000 tokens of the README (`BNK_NLL_SKIP=2000`), with the plain model's top-64
log-probabilities per row as the reference (`BNK_NLL_DUMP` / `BNK_NLL_REF`): the NLL, and the mean KL from the
plain model. For scale: the tensor-core prompt path of 2026-10-15 sits at KL 0.028 and was kept; 8-bit KV at
0.001; two plain runs of a prompt differ from each other by about 0.008 (near-ties).

| cold file | residency | NLL (plain 1.4432) | KL mean | same top-1 | verdict |
|---|---|---|---|---|---|
| machinery check: coldest 10% at Q8_0 (gate/up and down) | | 1.4432 | 0.008 | 96.4% | the cold path is correct: a near-lossless band changes nothing |
| iq2_s/q2_0:30, iq3_s/q2_0:40 (Q2_0 down in every layer) | 100% / 100% | 1.5516 | 0.247 | 81.5% | rejected: 2-bit down matrices in every layer cost a tenth of the perplexity |
| iq2_s/iq4_nl:30, iq3_s/iq4_nl:40, Q2_0 down only in the 9 layers the IQ3_S release uses it | 92% / 94% | 1.4688 | 0.108 | 86.9% | rejected: the 2-3 bit gate/up without an importance matrix still cost four times the accepted band |
| the same 70% borrowed from the IQ3_S release's own blocks (`--borrow`: its per-layer IQ2_S / IQ3_XXS / IQ3_S gate/up, IQ4_NL / Q2_0 down, data-aware quantization) | 95.7% / 93.5% | 1.4511 | 0.069 | 89.9% | **between FP4-Local and the IQ3_S release, closer to FP4** (next row) |
| calibration: the IQ3_S release itself on the same text | 100% (its own file) | 1.4571 | 0.118 | 86.0% | the accepted "IQ3-class" model: a KL of 0.1 against FP4-Local is what different weights cost, not an error |

So the yardstick for a mixed file is not the 0.03 of a same-weights kernel change but the 0.12 of the IQ3_S
release you already run: the borrowed mix sits at half that, with NLL 0.5% above FP4-Local against the release's
1.0%. The borrowed file's speed: 3K read 1,633 tok/s (+47%), code decode 105 (+0.5%), prose decode 60.7 (-35%):
the prose run's 36 rounds of 8-row lookup windows run the IQ2_S / IQ3_XXS expert kernels at 8 rows per block,
which is far slower than NVFP4's; measured next on the IQ3_S release itself (draft 4 against 7).

With lookahead off, prose decode is plain 83.6 tok/s against borrowed 80.5 (verify 25.1 -> 27.0 ms: the band's
extra gate/up and down launches, ~2 ms per forward; misses 2.9% -> 1.3%). The 61 tok/s above was that cost
compounded by lookahead (its speculative forwards carry the same extra launches, and only 14% of them were
confirmed in that run). Two fixes are open: one launch per phase over every region (a per-block format switch
instead of a launch per band), which removes the 2 ms, and a lookahead gate that counts the cost of a miss.

**With one launch per phase** (`moe_hits_regions`), the borrowed file against plain FP4-Local, `bnk run`, greedy:

| | plain | hybrid | |
|---|---|---|---|
| 3K-token read | 1,118-1,126 tok/s | 1,607-1,640 | +45% |
| prose decode, lookahead off / on | 87.3 / 86.4 tok/s | 88.1 / 94.0 | +1% / +9% |
| code decode, lookahead off / on | 114.3 / 111.7 tok/s | 135.1 / 133.9 | +18% / +20% |
| expert misses, prose / code | 2.9% / 7.1% | 1.3% / 2.2% | |
| CPU experts per forward, code | 8.7-9.9 ms | 3.1-3.5 ms | |

`tools/park_check.py` (3 slots) and `tools/serve_check.py` pass with the file loaded (`--cold-experts`, and
`--cache-gib 22` so a static cache holds the bands; with their default zero cache the engine logs that the bands
do not fit and runs plain). A cache too small for the bands never throws: the file is ignored for that run.

The second file's speed: 3K read 1,565 tok/s (+41%), code decode 122 tok/s (+17%), prose decode 86.9 (-7%: the
extra gate/up and down launches per band, about 2 ms per forward, on a round with nothing else to gain), misses
2-3%. Speed with the first file, for what the mechanism gives at full residency: 3K-token read 1,108 -> 1,789
tok/s, decode on code 104.5 -> 130.0 tok/s (the 12 ms of CPU experts per forward gone), prose decode unchanged
(93.8 -> 92.7), a 50-token turn 462 -> 341 ms, misses 0.

## Item 4 beside it: draft 7 where the window is free

The configs of the IQ3_S release and the 27B now draft 7 tokens per round (`speculation.draft`; FP4-Local stays at
4 until its CPU misses are gone). Measured on the IQ3_S release (`bnk run`, greedy): a verbatim file rewrite
261.5 -> 274.5 tok/s (6.81 -> 7.31 tokens per round), prose 102.4 -> 102.5 (2.46 -> 2.49 per round), the verify
round 18.4 -> 18.9 ms on the copy task and 21.6 -> 22.0 on prose: the 8-row window costs under a millisecond with
every expert resident, as predicted. On the Q8_0 27B the README prompt went 75 -> 80 tok/s and the copy task
stayed at its ~200.

## Beside it: the RCO IQ3_S 27B

`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` (the `-mtp` file, with the drafter block) runs in the dense engine after two
loader changes: the 14 small Q2_K / IQ1_M tensors and the BF16 `ssm_alpha` / `ssm_beta` matrices become Q8_0 at
load (the Q8_0 release ships them so). Against the Q8_0 27B on the same prompts: decode 81 against 80 tok/s on
the README, 202 against 200 on the copy task, reads 1,280 both; NLL 1.4352 against 1.4166 (+1.3%), KL 0.039,
same top token 91%. Half the bytes per forward bought no speed: the IQ-format GEMVs are bound by their unpack
instructions, not by HBM. What it buys is VRAM (6 GB per card instead of 13), hence fp16 KV in its config.
