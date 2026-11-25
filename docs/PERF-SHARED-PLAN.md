# Shared performance work, DeepSeek first (branch `perf-shared`, 2026-11-25)

The ideas list in `docs/DEEPSEEK-NEXT.md` ("Ideas list (2026-11-25)"), ordered so each step lands in code the
engines share where it can: the expert cache (`src/core/expert_cache`) and the CPU expert pool (`src/cpu`) serve
both MoE engines (DeepSeek and the Qwen engine: Flash-Next IQ3_S, CYBER-FROST); the dense engine (27B) has its own
tensor-parallel path.

| Step | What | DeepSeek | Other models |
| --- | --- | --- | --- |
| 1 | `tools/gpu_busy.py`: each GPU's busy and memory share in a request's prompt read and decode (NVML samples) | what is idle | every engine |
| 2 | Warm start: the resident experts saved beside the counts file, loaded first | restart 20 -> ~30 tok/s (to verify) | Qwen MoE engine |
| 3 | CPU pool: SMT siblings without spinning; faster one-token IQ2 dots | ~20 ms a token on the CPU | the same pool |
| 4 | Two DeepSeek slots in two lanes (`forward_pipe`) | two agents at once | (Qwen has it) |
| 5 | Disk parking for the Qwen engines; parking for the dense engine | done in v0.6.1 | Qwen MoE, 27B |
| 6 | DSpark's experts in VRAM, on the idle card | cheaper draft rounds | |
| 7 | 27B exchanges in fp16 / peer stores | | 27B |
| 8 | GPU 1 on a prompt's skipped groups (`PREFILL-SKIP-PLAN.md` 2b) | GPU 1 idles in prompt reads | |

## Step 1: measured (2026-11-25, production ds-xs server, master before v0.6.1)

`tools/gpu_busy.py --prompt-chars 20000 --max-tokens 300`: 6,475-token prompt in 5.5 s, 300 tokens at 28.2 tok/s.

| Phase | GPU 0 busy | GPU 0 memory | GPU 1 busy | GPU 1 memory |
| --- | --- | --- | --- | --- |
| Prompt read | 84% | 24% | **12%** | 1.4% |
| Decode | **40%** | 6.4% | **40%** | 9.8% |

- In a 6K prompt read GPU 1 is nearly idle: the long-prompt skip leaves the second stage only the tail group. So step
  8 matters at ordinary agent sizes, not only at 70K.
- In decode each card is idle ~60% of the time (busy counts any kernel, so a kernel waiting on the CPU's answer
  counts as busy: 40% is an upper bound). Memory traffic is low: the cards wait on the CPU experts and on each other.

## Step 2: warm start

`<counts>.resident` (BNKR v1: n_layer, n_expert, then (layer, expert) pairs hottest first by the adaptive score) is
written with the counts (every 60 s while serving, and at quit). At load, `warm_ranking` puts it ahead of the counts'
ranking (`ExpertCache::init` drops repeats and completes the rest), so the cache starts where the last run left it.
`BNK_WARM_START=0` turns it off. Both MoE engines. Test: `build/test_expert_counts`. To measure: decode right after a
restart, warm vs `BNK_WARM_START=0`.
