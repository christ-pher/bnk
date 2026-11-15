# Bug report and verification: two requests sent at once serialized the server (fix `a1d9c8d`)

## The symptom

Under multi-agent load the server, after running for a while, stopped batching conversations and served
everything one at a time, even though the engine ran with `--slots 3` and its own log confirmed
"3 conversation slots". New requests queued behind one another for hours.

## What the logs showed (run `~/.cache/bnk/server-8181.log`, 2026-11-02 21:54 -> 11-04 04:49, 2,665 requests)

- The engine was busy ~24 of ~31 hours; wall-clock windows overlapped at concurrency 1 for 18.3 h,
  at 2 for 5.6 h, and at 3 for 0.03 h — with 3 configured slots, three-at-a-time essentially never happened.
- 2,551 of 2,664 requests arrived while the engine was already busy: the queue, not the GPU, was the limiter.
- Individual conversations sat in "reading" for hours (a prompt was logged with `▶` and its `✓` never came);
  the report that prompted this investigation saw a pair of prompts sent 140 ms apart stuck together.

Note the log cannot show engine-side `active` over time (telemetry goes to the Python side and is not
persisted), so these HTTP-level windows include queue time; they localize the decay to the Python layer but
do not prove which counter leaked.

## Root cause (found in `serve/engine.py`, fixed by `a1d9c8d`)

1. The engine child is started with `bufsize=0`, so `_send` wrote each request line straight to the pipe.
   A pipe write larger than `PIPE_BUF` (4 KiB) is not atomic on Linux: two HTTP threads sending huge
   prompts (hundreds of KB of JSON token arrays) block after the 64 KiB pipe buffer and their remaining
   bytes interleave. One 360 KB line is a `write()` that the pipe cannot hold at once, so both lines split.
2. The engine then parses neither line and answers `bad request: ...` (`src/server/serve.cpp:313`) — an
   error with no `id`. `_read_out` routes an id-less message to the `"_misc"` queue, which nobody has, and
   drops it. Neither request thread ever learns its request died.
3. Those threads wait forever in the old `q.get()` (no timeout). The `generate()` generator never reaches
   its `finally`, so `_release()` never runs and `_running` keeps counting the dead requests.
4. With `--slots 3` and two leaked turns, `_acquire` admits at most one live request at a time:
   `self._running >= self.slots()` leaves a single slot. This is exactly the symptom — a slow crawl from
   full batching to one-at-a-time, one pair of colliding requests at a time, never recovered until restart.

The corruption needs two big prompts to hit the pipe at the same instant, which is why it appeared "after a
while" under agent traffic: the longer the server runs, the more likely two agents' turns overlap.

## Empirical verification (fake line-protocol engine + the real `Engine` class)

A fake engine that prints `ready {"slots":3}`, answers `bad request` without an id for unparsable lines
(exactly `serve.cpp:313`), and completes valid requests with `done`, driven by nine ~360 KB concurrent
generate calls started within ~140 ms of each other:

- **Before (`a1d9c8d^`):** two requests were never parsed; their threads blocked in `q.get()` forever,
  `_running` leaked to 2 of 3, and three fresh concurrent requests peaked at one real slot. The two
  corrupt threads never exited, matching the "stuck in reading for hours" observation.
- **After (`a1d9c8d`):** 9/9 requests completed, `_running` returned to 0.

Test harness (kept out of the repo; recreate under `/tmp`): run `drive.py ENGINE_PY` against the fake
engine, where `ENGINE_PY` points at `git show HEAD~1:serve/engine.py` or the current file. Note the harness
threads are non-daemon by default, so a buggy build hangs the harness at exit — that hang is itself the
leak's signature.

## The fix

1. `_send` takes `self.wlock` around the whole `write + flush` of one line: one complete line at a time,
   whatever other thread is sending. It is the only stdin writer (`generate`, `cancel`, `quit` all route
   through `_send`), and the engine's reader thread drains stdin continuously, so the lock is never held long.
2. `generate`'s `q.get(timeout=1800)` breaks any future lost-answer case: after half an hour of silence for
   that rid it sends `cancel` and raises `EngineError` instead of holding a turn slot forever.

Verification suite: `tests/test_serve_run.py` (skipped without a DeepSeek GGUF), `tests/test_loop_guard.py`
and `tests/test_tool_calls.py` all pass with `.venv/bin/python tests/<name>.py`. The fixed server has been
live since 2026-11-15 13:32 (port 8181, slots 3).

## Residual caveats (minor, not blocking)

- The 1800 s timeout also counts time a request spends in the engine's queue before admission, because the
  first event with its rid comes at prefill start. A fourth request queued behind three legitimate
  32,768-token generations could in principle wait past 30 minutes (3 x 32768 tokens at batched ~17 tok/s
  is ~1,640 s each) and be cancelled spuriously. At the observed 50-90 tok/s the margin is 6-20x, but the
  corner is real; an engine-side "queued/ack" event at admission would close it.
- A `cancel` for a rid the engine never saw inserts the id into C++ `cancelled` (`src/server/serve.cpp:324`),
  which is only erased when a matching request finishes (`serve.cpp:395`): unknown ids accumulate forever.
  It is bytes, and it only happens once the safety net fires.

## Notes from the fixer (2026-11-15, same day)

- The first live evidence was a single `/api/stats` snapshot taken while the bug was active, before the fix
  existed: `live.active: 1, slots: 3, waiting: 4`, and conversations #13, #16 stuck with `status: "reading"`
  and `since` timestamps 140 ms apart (1794723913.42 / .56) while later requests in other conversations
  kept completing. That rules out the two engine-side serializers the snapshot also showed (VRAM parking and
  the `--slice` turn-taking): the engine was healthy and admitting, only those two rids were never answered.
  Telemetry is not persisted, so this snapshot is as close to a trace of the leak as it gets; if a similar
  decay is ever suspected again, capture `/api/stats` first.
- The `"_misc"` drop in `_read_out` is a general observability hole independent of this bug: any engine error
  that arrives without an `id` (the `bad request` line being the main one) vanishes silently. One line in
  `_read_out` appending such messages to `log_tail` would have made this incident visible in the dashboard's
  log panel the moment it happened, and it costs nothing.
- The queued-before-admission corner in the timeout caveat can be split without a protocol change: the loop
  already tracks `started` (the first event for the rid). A short silence limit once events are flowing
  (the engine emits telemetry/prefill progress for the admitted request it is working on; a lost rid goes
  silent immediately) and a long one before the first event would keep the safety net sharp without ever
  cancelling a legitimately queued request.
- Suggest making the harness a real test (`tests/test_engine_send_race.py`): it needs no GGUF, runs in
  seconds, and pins exactly this bug — the fake engine's `bad request`-without-id behavior should stay
  line-for-line identical to `serve.cpp:313`. The non-daemon-thread hang is a feature there: `timeout 60`
  around it turns the leak signature into a test failure.

