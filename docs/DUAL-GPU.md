# Two GPUs (a pipeline over several V100s)

bnk runs on several GPUs as a **pipeline**: the model's 48 layers are split into contiguous ranges, one per GPU.
Each GPU holds its layers' dense weights, their routed experts (as many as fit), their KV cache and recurrent
state; the first GPU also embeds the tokens, the last one holds the output head and the MTP drafter. A forward
runs the first GPU's layers, hands the residual rows to the next GPU, and so on.

Why a layer split: the model does not fit one V100. IQ3_S is 3.9 GiB of dense weights, 46.8 GiB of experts and a
1 GiB drafter; one 32 GB card keeps ~43% of the experts and the CPU computes the rest (decode is bound by those
CPU experts). Two cards hold **every expert** (~25 GiB each), so the CPU leaves the decode path. Only a few
hundred KB cross between the GPUs per forward (the residual rows), so plain PCIe without NVLink or peer-to-peer
access is enough. Splitting each layer across both GPUs instead (tensor or expert parallelism) would need two
exchanges per layer, ~100 per forward, each a PCIe round trip.

## Running it

Nothing to configure: the server uses every GPU it can see (`--gpus all`, sm_70 or newer). With two V100s:

```bash
./run.sh                      # both GPUs
BNK_GPUS=0 ./run.sh           # GPU 0 only, as before
BNK_GPUS=0,1 BNK_SPLIT_LAYERS=25 ./run.sh   # explicit GPUs and split (layers 0-24 on GPU 0)
```

The engine prints the plan when it starts:

```
bnk: 2 GPUs, layers split 0-24 on GPU 0 (all resident: 27.9 of 29.6 GiB), 25-47 on GPU 1 (all resident: 28.4 of 29.6 GiB)
bnk: stage 1 of 2: GPU 0, layers 0-24
...
bnk: VRAM expert cache for layers 0-24: 12800 of 12800 experts (100.0%), 24.4 GiB
```

The split balances what each GPU would hold with every expert resident and the KV of a full context (plus the
embeddings on the first GPU, the head and drafter on the last) against each GPU's free memory. "All resident"
is that estimate; the expert cache then takes whatever each GPU really has left, and gives it back to the context
(KV) as conversations grow, as on one GPU.

Options (`bnk serve` / `bnk run`, or `run.sh` / the server, which pass them on):

| Option | Server / env | What |
|---|---|---|
| `--gpus 0,1` | `--gpus`, `BNK_GPUS` | CUDA devices, one pipeline stage each, in layer order. `all` (default): every sm_70+ GPU |
| `--split 25` | `--split`, `BNK_SPLIT_LAYERS` | the first layer of each GPU after the first (default: balanced by memory) |
| `--cache-gib 20,24` | config `engine_args` | expert cache size per GPU (one value: the same cap on each) |
| `--vram-reserve-gib 1.5` | config `engine_args` | VRAM each GPU keeps free beyond the engine's budget (default 0.75) |

GPU numbers follow `nvidia-smi` (the engine sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`).

**Sharing a GPU with Plex.** NVENC transcoding needs a few hundred MiB of VRAM on the card it uses. bnk keeps 0.75
GiB free on each GPU and re-reads the driver's free memory whenever it maps more, so a transcode that starts
while bnk runs takes its memory out of bnk's expert cache, not the other way round; a transcode that needs more
than the reserve can fail to start if bnk has just filled the card. To leave Plex guaranteed room, raise the
reserve (`--vram-reserve-gib 1.5`) or cap the expert cache on the GPU Plex uses (`--cache-gib 22,24`).

## How it works

* **Stages.** `Engine` is the first stage and owns the next (`next_`); each stage is a full engine restricted to
  its layers `[l0, l1)` on its device: its own streams, CUDA graphs, VRAM budget, expert cache, KV and recurrent
  state, prompt buffers. Every public engine call made on the first stage covers the whole pipeline: a forward
  runs every stage, logits / residuals / the drafter come from the last stage, `select` / `reset` / `commit` /
  snapshots / context growth apply to every stage. Each stage runs its CUDA work under a `DeviceScope`.
* **Shared host side.** The routed experts in pinned host RAM (`ExpertStore`, registered portable) and the CPU
  expert pool are the first stage's and serve every stage: the 47 GiB store exists once.
* **A decode forward** prepares every stage first (books the windows, captures a graph for a new layout), then
  queues the graphs: the second stage's stream waits on an event the first records after its graph. The host
  then answers each stage's CPU experts in layer order while the GPUs run. Nothing waits on a queued graph
  before its CPU part has been answered (a capture can synchronize its stream, so captures come first; the other
  order deadlocked).
* **Between stages** the residual rows (4 x 2560 floats per token, 40 KiB) go through pinned host memory: the
  first graph ends with a device-to-host copy, the next one starts with a host-to-device copy. Prompt chunks
  have their own pair of buffers (up to 8,192 rows each). Peer copies work since the host's IOMMU went to
  passthrough (2026-10-19, docs/DENSE.md) but are not used here: one handoff per forward saves ~15 us of a
  10-20 ms decode forward, and a 2,048-token chunk's 84 MB (~8 ms each way) overlaps ~4 s of compute, while
  device-side prompt buffers would take ~0.7 GB from the second GPU's expert cache.
* **Prompts are pipelined.** While the first GPU reads chunk k+1, a worker thread runs chunk k through the later
  GPUs (and feeds the drafter's KV). Between chunks both are idle, and the server's other conversations decode
  then, as before. `BNK_PIPE_PROMPT=0` reads chunk by chunk through all GPUs.
* **Parking** is decided once (on the first stage, with the bytes of every stage counted against `--park-gib`)
  and carried out on every stage: each holds its part of a parked conversation, under the same id. When a GPU
  runs short of VRAM, idle conversations are parked off every GPU.
* **Routing counts** (`--counts`): each stage counts its own layers and adds them to the file in turn.

## Checking it on one GPU

The same GPU listed twice runs both stages on one card, which exercises everything but the second device:

```bash
build/bnk run --model M.gguf --tokens-file p.txt --gpus 0,0 --split 24 --cache-gib 9   # two stages on GPU 0
.venv/bin/python tools/park_check.py M.gguf --mtp ... --gpus 0,0 --slots 2
.venv/bin/python tools/serve_check.py M.gguf a.txt b.txt c.txt --mtp ... --gpus 0,0
```

Cap the expert cache (`--cache-gib`): the first stage would otherwise take the whole card before the second
loads.

## Measured on two V100s (2026-10-14)

Two Tesla PG500-216 (V100 SXM2 32 GB on PCIe adapters, 150 W cap, PCIe 3.0 x16 each, no NVLink or peer access).
IQ3_S, every expert resident (layers 0-25 / 26-47, 24.8 + 23.2 GiB of experts).

| | One V100 | Two V100s, first run | Two V100s, after the GPU work below |
|---|---|---|---|
| GPU time per forward, MTP windows (~3 rows) | | 16.45 + 14.76 ms | 11.88 + 10.77 ms (-27%) |
| Decode, one conversation, plain greedy | ~45-50 tok/s | 54.7 tok/s | 69 tok/s |
| Decode, one conversation, MTP (`bnk run`) | ~60-65 tok/s | 67-78 tok/s | ~100-120 tok/s (acceptance-dependent) |
| Server, web UI settings (temp 0.7, 2,048 tokens) | | 76-79 tok/s | 92-96 tok/s |
| 2 / 3 conversations together (two lanes) | | 83 / 131 tok/s (batched) | 137 / 173 tok/s |
| Prompt reads, 36-56K tokens | 570-750 tok/s | 1,800-2,020 tok/s | 1,985-2,107 tok/s |

No throttling seen: 48-57 °C under load, the power cap reached briefly while reading prompts.

What changed (each its own commit on `dual-v100`, measured alone; `docs/ROADMAP.md` item 8 has the details):

* Dense GEMVs on the tensor cores (`gemv_mma_k`, `mma.m8n8k4`) from 4 rows, BF16 matrices as F16, no loads for
  padding tokens, K split per row width: 8-row GEMVs 1.5-2x faster, closer to fp32 (error 5e-3 -> 3e-4).
* Routed experts: IQ grid tables in shared memory, activations through L1, each expert's own token count per
  block, 8 lanes per row for several tokens: IQ2_S layers 245 -> 179 us at 5 tokens.
* Small kernels: the HC norm and mix quantize their own outputs, `hc_combine` folded into the next norm,
  `route_topk` + `moe_plan` in one launch, `moe_wait` folded into `moe_reduce`, vectorized norms, the DeltaNet
  recurrence over 4 warps, independent work on side streams (shared expert, gate / beta / alpha, indexer, inject).
* Drafter: draft head re-quantized to Q4_K (step 0.82 -> 0.72 ms, same acceptance).
* Two lanes on by default.

Where a forward goes now (nsys, ~3-row windows): dense GEMVs ~49% (tensor-core kernel at ~400-450 GB/s for big
matrices, the limit is load latency at 50% occupancy), routed experts ~25% (~450 GB/s; IQ2_S decode is
instruction-heavy), the rest ~25% (DeltaNet recurrence, attention, routing, norms).

**Exactness.** Speed-tuned kernel choices depend on the window's row count (tensor cores from 4 rows; 16 or 8 lanes
per expert row), so a conversation decoded beside others can differ from itself alone in near-ties.
`BNK_EXACT=1` pins both choices (~1% slower with MTP, 11% plain): then batched and two-lane decoding equal
decoding alone token for token (3 conversations x 200 tokens), and the MoE adds a token's experts in its own routing
order in every mode. `serve_check` and `park_check` set it; both pass on two GPUs.

Sampling note: a request with `top_k: 0` used to take the host sampler (~23 ms per row, decode 63 -> 26 tok/s); it
now samples from the GPU's 1,024 most likely tokens and each row's normalizer (identical distribution).

## What to expect (projected on one V100, 2026-10-09)

The second V100 had not arrived yet, so the speed of two GPUs is projected from one: two stages on the same card
(`--gpus 0,0 --split 20 --cache-gib 21,0`), the first holding layers 0-19 with **every** expert resident (18.85
GiB), and each stage timing its own part (`bnk run` / `bnk multi` print it per stage). A fully resident stage is
what each GPU of a real pair runs; the second stage here (CPU experts) only completes the forward.

| Layers 0-19, every expert in VRAM | Measured | Whole model on two GPUs (projected) | One GPU today |
|---|---|---|---|
| Decode forward, one conversation (~3.5 rows) | 10.8 ms | ~27 ms (~0.54 ms per layer + head) | 45-70 ms (CPU experts) |
| Decode forward, 3 conversations (8 rows) | 17.7 ms | ~44 ms (~0.89 ms per layer) | ~55-65 ms |
| Prompt chunks of 2,048 tokens (9.8K prompt) | 2,404 tok/s | ~2,000 tok/s (24 layers per GPU, overlapped) | 570 tok/s (750 with 8K chunks) |

So, roughly: **one conversation ~85-90 tok/s** greedy at short context (65 on one GPU), **three conversations ~105
tok/s** together (85-90), and **prompt reads ~3x faster** (every expert is already on a GPU, nothing streams over
PCIe, and the two GPUs work on consecutive chunks at once). Decoding no longer waits for the CPU; the GPU's own
per-layer time (many small kernels: ~0.54 ms per layer where the weights alone would take ~0.17 ms) is the next
limit, and it is the same on one GPU.

A layer pipeline does not make one conversation's forward faster than one GPU holding everything would: the two
GPUs take turns. Batched conversations share the turns the same way, unless they decode in two lanes (below).

## Two lanes: both GPUs busy while several conversations decode (`BNK_PIPE_DECODE`, on by default)

With two GPUs and at least two requests in flight, `BNK_PIPE_DECODE=1` splits the requests into two lanes and
keeps one batch of each in flight: every step runs one lane's batch on GPU 0 while the other lane's batch (which
left GPU 0 the step before) runs on GPU 1, then verifies, commits and drafts that one. Each lane gets up to 8 rows
of its own (more drafts verified per conversation than when all of them share 8 rows), and neither GPU waits for
the other's layers.

```
GPU 0:  A1 ─────── B1 ─────── A2 ─────── B2 ───────
GPU 1:  ·········· A1 ── fin ─ B1 ── fin ─ A2 ── fin ─
```

From the measured stage times (13 ms per GPU for one conversation's rows, ~3.5 ms to verify, commit and draft),
two conversations would go from ~45 ms per round batched to ~33 ms per two lane steps, ~+35%; more for three (each
lane keeps 8 rows). **Measured on two V100s (2026-10-14, MTP, greedy, `bnk multi --gpus 0,1 [--pipe]`):** two conversations 83 -> 106
tok/s together (+28%), three 131 -> 159 tok/s (+21%); each conversation ~53 tok/s instead of 41-44. Each
conversation's tokens in two lanes are exactly its tokens decoded alone (`--solo`); batched rounds are not quite
(with GPU experts the expert sums follow the batch's routing), so two lanes are also the more exact mode. **On by
default** when the two stages are on different GPUs; `BNK_PIPE_DECODE=0` turns it off (and `=1` forces it, e.g.
for `--gpus 0,0`). Through the HTTP server: concurrent streams at 50-61 tok/s each, a client disconnecting
mid-stream and follow-up requests complete cleanly; `serve_check` passes.

What makes it correct: a verify window's DeltaNet intermediates stay on GPU 0 until the window is committed, and
GPU 0 runs the other lane's forward in between; each lane has its own window buffers, hand-off buffers and commit
graphs, so the commit replays its own lane's rows. A new request (its prompt is read on both GPUs), or a lane
running out of requests, first drains the pipeline (the batch in flight finishes on GPU 1), and the server goes
back to one batch per round. Checked on one card: `bnk multi --pipe` gives the same tokens as batched decoding
(three conversations, deterministic CPU-only experts), `serve_check` passes with it on, and an HTTP run with three
streams, a client disconnecting mid-stream and follow-up requests completes cleanly.

## Verified (on one GPU, two stages)

* Greedy output equal to the single-GPU engine (and to master): 96 tokens after a short prompt, 64 after a
  9.8K-token prompt (chunked prompt path), with the prompt chunks pipelined and without.
* `tools/park_check.py --gpus 0,0 --slots 2`: a conversation parked and brought back produces the same tokens
  (history extended, and re-rendered from a snapshot). PASS.
* `tools/serve_check.py --gpus 0,0`: three requests served together (batched rounds, decoding between another
  request's pipelined prompt chunks) produce exactly their tokens alone. PASS.
* The same with `BNK_PIPE_DECODE=1` (two lanes). PASS. `bnk multi --pipe`: tokens identical to batched.
* A 72K-token prompt (chunks of 8,192, pipelined): greedy output equal to master.
* VRAM pressure (a park_check variant with `--vram-reserve-gib` raised until two ~21-36K conversations no longer
  fit together): the idle one is parked off both stages to make room, later the other is parked to bring it back,
  and its next turn is identical to a run without the other.
* Through the HTTP server: concurrent streaming requests, a client disconnecting mid-stream, follow-up requests,
  with and without `BNK_PIPE_DECODE=1`; telemetry lists both stages.
* One GPU (`--gpus` unset or one id) produces exactly master's tokens.

Not verified, because it needs the second card: anything specific to two devices (cross-device event waits,
per-device memory, the split planner's real numbers). The code keeps every stage's CUDA work under its own device
(`DeviceScope`) and shares only portable pinned host memory between them.
