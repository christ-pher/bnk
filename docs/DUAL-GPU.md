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
  have their own pair of buffers (up to 8,192 rows each).
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
