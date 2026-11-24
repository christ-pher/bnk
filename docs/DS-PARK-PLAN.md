# DeepSeek parking and prompt states on disk (branch ds-park-disk, 2026-11-24)

Borrowed from TensorFold (host KV tier `TF_GLM_KV_HOST` / `TF_GLM_HOST_CACHE`, disk cache `--spill-gib`), on top
of what the Qwen engine already does in host RAM (`src/engine/park.cpp`).

**Problem.** `DsEngine` runs one slot and has no parking (`parked()` is 0), so two agents taking turns evict each
other and each re-reads its whole prompt (190K tokens: 260-276 s). Nothing survives a restart either: a 256K cold
read is 124-512 s against 1-2 s from NVMe.

## What a DeepSeek conversation's state is (per pipeline stage)

| Part | Size | Per position? |
| --- | --- | --- |
| history (token ids, image cells as `-1 - hash`) | 4 B / token | yes |
| compressed latents `ckc` and index keys `ikc`, kv-source layers | rows = pos / ratio, fp16 x (512 + 128) | yes: rows past the cut are rewritten |
| window ring, every layer | 136 x 512 fp16 (139 KiB a layer) | no: slot = pos % 136 |
| compressor's open group `st_kv` / `st_sc` (ratio > 1) | (ratio - 1) x 512 f32 x 2 | no |
| snapshots (host): the open group | as above | no |
| DSpark (last stage): its rings per stage + `fed_lo/fed_hi` | 3 x 136 x 512 fp16 | ring |

## Phase 0: snapshots must keep the window ring (an existing bug)

`checkpoint_here` saves only the open group, and `rollback_here` restores only that. The ring is read with no
validity check (`ds_prefill.cpp` `ds_ring_read`, the decode window attention). After a rollback past more than 8
generated tokens, the slots for the 128 positions before the cut hold later positions. That is every new user turn:
the template drops the last turn's reasoning (`drop_thinking`), so the prompt diverges after the previous prompt's
end and the engine rolls back to that snapshot. DSpark already guards its own ring (`DsSpark::restart`).

Fix: a snapshot also copies every layer's ring (D2H, ~6 MB for 43 layers) and a rollback brings it back.
Check: greedy, turn 1 then a turn-2 prompt that rolls back to turn 1's prompt end. The tokens must match a fresh
engine reading turn 2 from scratch (`BNK_DS_NO_REUSE` style reset), and must differ before the fix.

## Phase 1: parking in host RAM

The Qwen design, without slot forking (one slot):

- `Parked { history, ckpts, host, cap, bytes, used }` per stage. `host` holds `ckc`/`ikc` rows up to pos, the ring,
  the open group and (last stage) the DSpark rings and `fed_lo/fed_hi`.
- `select_conversation(prompt)`: the best of live / parked by `reuse_of`. The live conversation is parked when the
  request would lose at least `park_min` tokens of it. A parked one the prompt extends is restored (the entry
  goes). A parked one that shares only a prefix with the prompt is copied out up to its best snapshot
  (`fork_parked`: rows up to r plus the snapshot's state and ring), and the entry stays, so two agents over the
  same repo dump share the read.
- Budget `--park-gib` (server default 24). The DeepSeek engine also keeps `BNK_DS_PARK_FREE_GIB` (default 16) of
  MemAvailable free, since the Engram tables live in the page cache and the expert pin already holds ~100 GiB.
- `parked()`, `parked_bytes()`, `drop_parked_all()`, `park_stats` as on the Qwen engine (dashboard, /stats).

## Phase 2: on disk

- Directory `<cache-dir>/park/<identity>/`: identity = a hash of the model files' names and sizes, the stage
  layout (layer split), the mmproj, and a format version. A file is one conversation: a header (magic, version,
  history, snapshot positions, per-stage section sizes) then each stage's bytes, as parked.
- Written when an entry leaves RAM by LRU (instead of being dropped), and at shutdown for the live conversation and
  every parked one (`drop_parked_all` is a clean slate, so it deletes them too). Written by a worker thread from
  the pinned copy, so a park does not wait on the disk.
- At start, the headers are indexed (history only). `select_conversation` scores disk entries like parked ones. A
  hit is read into pinned RAM (direct reads, ~2-3 GB/s) and then restored as a parked entry.
- Budget `BNK_PARK_DISK_GIB` (default 32), oldest first out. 109 GB free on the disk today.
- The Qwen engine can use the same file layer later. Not in this branch.

## Measures

- Two agents alternating on ds-xs at 50K and 150K: the turn's prompt time before vs after (expect seconds -> ~0.5 s).
- A restart in between: the first turn's prompt time (cold read vs disk resume).
- Quality: greedy, a resumed turn's tokens equal the never-parked run's.
- Decode speed unchanged (the parking is outside the forward).

## Results (2026-11-24, ds-xs on the two V100s)

Three bugs found and fixed on the way, two of them older than this branch:

1. **Snapshots without the window ring** (phase 0 above). Before: turn 2 after a 300-token reply decoded differently
   from a fresh read of the same prompt; after: identical.
2. **No snapshots at all for DeepSeek.** `serve/server.py` placed them at `<|im_start|>`, which this tokenizer lacks,
   so every new user turn read the whole prompt again (119K tokens: 134 s), parked or not. Now before the last
   `<｜Assistant｜>` and the first `<｜User｜>`.
3. **Ctrl-C / SIGTERM killed the engine before the server's quit** (the signal reaches the whole process group), so
   nothing reached the disk. The engine now ends its loop on the first SIGINT / SIGTERM and persists.

Exactness (`tools/ds_park_check.py --poison`): the engine is not bit-deterministic run to run (first-token KL
3e-3..5e-2 between identical runs), so a check compares logits against the spread of three reference runs, with the
live slot filled with NaN before every restore. extend 2.0e-2 / rerender 8.4e-3 / disk 2.2e-2, all within the
noise; the control (open groups left NaN, `BNK_DS_PARK_BREAK=1`) 13.8.

Two agents taking turns (`tools/multi_agent_bench.py 2 300000 3 300`, time slices on):

| | first turn (cold) | later turns: read | later turns: whole turn | decode |
| --- | --- | --- | --- | --- |
| before (master behaviour) | 125-131 s for 114-119K | 125-134 s (reused 0) | ~145 s | 26-28 tok/s |
| this branch | 125-131 s | **2.0-5.6 s** (reused all but the new ~300-500 tokens) | **28-31 s** | 28.6-29.4 tok/s |

Parking a 119K-token conversation: 0.37 GiB (3.2 KB a token), 20-550 ms (a reused spare buffer is the fast case);
restore 17-20 ms.

Across a restart (one agent at 119K tokens; SIGTERM to the server's process group, then a fresh start): the stop
wrote 7 conversations (2.6 GiB) in 2.9 s, the start found them, and the next turn reused 119,399 of 119,415 tokens
and read for **1.0 s** (437 ms of it the disk read) against 132 s cold. That turn decoded at 20.1 tok/s against 29.9
before the restart, probably the expert cache adapting again after the start (not measured further).
