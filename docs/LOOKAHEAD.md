# Lookahead pipelining: one stream on both V100s (2026-10-19)

Idea C of `docs/PERF-IDEAS-2026-10-19.md`. On the two-GPU pipeline a decode round is GPU 0's 24 layers (~11 ms),
then GPU 1's (~10 ms) and the drafter; each card idles while the other works, and a single stream could not use
the idle card because the next window depends on this one's verdict. Lookahead runs the next window anyway, on
the assumption that every draft is accepted: when that holds (12% of prose rounds, half of code-writing
rounds, 85-92% of copy rounds, measured with `BNK_LA_PROBE=1` before building this), the round shrinks to GPU 1's
half. When it does not, nothing was touched.

## How a round goes

1. Window W = the pending token + its drafts goes through GPU 0 (lane L) and then GPU 1, as before.
2. While GPU 0 works, the drafter (on GPU 1, now on a stream of its own so its steps overlap the main model's
   kernels) chains ahead: the tokens that would follow W's last draft, as many as the next window would hold
   (`max_draft + 1`; a lookup match supplies them for free). This runs on a worker thread; the main thread keeps
   serving GPU 0's CPU experts.
3. When GPU 0 is done with W: its recurrent state (DeltaNet states, conv histories, the n-gram history) is copied
   to shadow buffers (~60 MB device to device), W is committed **in full** into the shadow, and the chained
   window W' runs through GPU 0 in the other lane against the shadow, its rows landing in that lane's hand-off
   buffer. The real state and W's intermediates are untouched. Meanwhile GPU 1 runs W.
4. Verdict. If every draft of W was accepted and the bonus token equals W'[0]: the shadow is copied back (the real
   state is now "W committed"), W' becomes GPU 0's pending window, GPU 1 commits W, and the next round only runs
   GPU 1 (W' from the hand-off buffer) while GPU 0 speculates W'' the same way. The drafter is fed W's rows and
   stepped through W' so its chain continues. Otherwise W' is discarded (its tokens dropped from the history; the
   KV cells it wrote beyond the committed position are overwritten by the real next window), and the round ends
   as before: commit the accepted rows, draft, next window.

The speculative window never changes what is emitted: the verdict comes from GPU 1's logits of the real window, and
a confirmed W' is exactly the window the normal path would have built. `BNK_LOOKAHEAD=0` turns it off (no shadow
buffers, the old rounds). `bnk run` and the server (one active request on a two-GPU pipeline) use it; batched
rounds and two-lane decoding (several requests) are unchanged, and a request whose next window is half done
finishes that round alone before another request's prompt or batch runs (`la_drain`).

## Code

- `src/engine/engine.cpp`: `forward_lookahead` (both stages launched, the chain on the `svc_` worker, then
  `spec_begin`: shadow copy, `enqueue_commit(cnt, shadow)`, the speculative forward prepared with the state
  pointers swapped (`swap_shadow`) so its graph bakes the shadow addresses, in the other lane), `spec_confirm`,
  `spec_discard`; `Seq::ssm_shadow / conv_shadow / ple_shadow / spec_*`; `fwd_prepare(..., spec)` and `fwd_end`
  record `spec_T` instead of `pending_T`.
- `src/engine/mtp.cpp`: the drafter's own stream (`after_main()` orders it after the main stream when it reads
  residual rows).
- `src/engine/generate.cpp`: `next_lookahead()` (the round above), `verify_window()` factored out of `finish()`,
  stats `la_rounds / la_hits`.
- `src/server/serve.cpp`: the single-request round and `la_drain`.

## Measured

`bnk run`, FP4-Local on the two V100s, greedy, draft 4, cutoff 0.8, 700 tokens decoded, lookahead on against off
(same binary, `BNK_LOOKAHEAD=0`):

| task | off | on | speculations confirmed | round, off -> on |
|---|---|---|---|---|
| rewrite a 1,000-token file verbatim (lookup drafts) | 176.6 tok/s | **209.3** (+18%) | 80 of 93 (86%) | 39.1 -> 23.8 ms verify |
| write a Python module (mixed, MTP drafts) | 111.7-126.2 (four runs) | 121.5-124.3 | 66 of 106 (62%) | 29-32 -> 24-25 ms |
| a 600-word essay | 98.4 | 94.3 | 5 of 18 (28%; the gate keeps the rest out) | 20.8 -> 21.0 ms |

Exactness: with window-independent kernels and a static expert cache (`BNK_EXACT=1 --adapt-every 0`) the copy
task's 600 tokens are identical with lookahead on and off (the hit path included: 84% of its rounds were
confirmed speculations). The essay differs from token 175 on, but so do two lookahead-off runs of it from each
other under the same flags (a near-tie this prompt reaches, decided by something outside the pinned kernel
choices), so that is not the lookahead's doing; its numbers above are therefore not the same text, and prose,
which rarely passes the gate, is unchanged within noise. Copy-heavy output, what agents produce when they
re-emit files, gains a fifth. `tools/serve_check.py` (a request decoding with lookahead while others' prompts
are read, then batched) and `tools/park_check.py` pass.

What limited the gain, in the order found: a missed speculation delayed the next round by the whole speculative
forward (fixed: the verdict is processed while it drains, so a miss costs only its tail); the extra graph
layouts of the other lane thrashed the 32-graph cache (48 with lookahead); after a confirmed speculation the
window's position was taken from a history that already held it, which made consecutive hits impossible and fed
the drafter at wrong cells (fixed: 45% -> 86% confirmed on the copy task). What remains: a speculative window
costs GPU 0 as much as a real one and its CPU experts compete with GPU 1's, so a miss is never free; the gate
(`BNK_LA_MIN`, the product of the drafter's probabilities over the window's drafts, default 0.8; lookup drafts
always qualify) decides when to try. Two more things the checks caught: a window without drafts must commit
inside its forward as it does on the plain path, and a request that ends after a confirmed speculation leaves
its next window half done on GPU 0 (`spec_abort` drops it when the request leaves, and `reset` clears it).

Trace switches for the next person: `BNK_LA_DEBUG=1` prints each round's phases with their times and the
verdict; `BNK_LA_SEQ_SVC=1` serves the second stage's CPU experts after the first's on one thread;
`BNK_MTP_OWN_STREAM=0` puts the drafter back on the stage's stream; `BNK_LA_PROBE=1` on the plain path counts
how often a speculation would have held.
