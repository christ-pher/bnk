# Prompt-lookup drafting and prefix forking (2026-10-19)

Items 3 and 10 of `docs/PERF-IDEAS-2026-10-19.md`: more accepted tokens per decode round, and conversations that
share a system prompt starting from each other's state instead of reading it again. Both engines get the
drafting; forking is the MoE engine's (the dense engine has no parking yet, item 9).

## Prompt-lookup drafting (`src/engine/generate.cpp`)

A verify window has 8 rows and a forward costs about the same for 1 row or 8, so every accepted draft is nearly
free. The MTP drafter proposes up to 4 tokens per round, one step each (~0.7 ms), at ~75% acceptance on prose
and near 100% when the model copies text. Prompt lookup proposes what followed the last occurrence of the
sequence's last 3 tokens, up to 7 at once, for nothing: agents re-emit file contents, identifiers, JSON
scaffolding and the text they quote.

- `LookupIndex`: the conversation's 3-grams in a hash table (n-gram -> the position after its latest
  occurrence), extended as tokens are committed; a request indexes its prompt once (~13 ms per 256K tokens).
- Every round the MTP drafter still runs (its K/V must follow the conversation) and gives its first draft; the
  lookup's chain is taken when its match has 6+ tokens of agreeing context before it, or when the MTP's first
  draft agrees with it, else the MTP's chain continues as before. `GenOptions::lookup_n / lookup_max /
  lookup_agree` (3 / 7 / 6; env `BNK_LOOKUP_N` 0 turns it off, `BNK_LOOKUP_MAX`, `BNK_LOOKUP_AGREE`).
- Each response's `timings` carries `lookup_rounds / lookup_drafted / lookup_accepted` beside `drafted` and
  `accepted`; `bnk run` prints both sources.

Measured (`bnk run`, two V100s, greedy, draft 4, cutoff 0.8; the 27B tensor parallel with 8-bit KV):

| model, task | lookup | tok/s | tokens per round | drafts accepted | per round |
|---|---|---|---|---|---|
| FP4-Local, rewrite a 1,000-token file verbatim | off | 150.5 | 4.94 | MTP 791 / 803 | verify 29.1 ms, draft 3.5 ms |
| | on | **174.8** | **7.22** | lookup 805 / 868 (124 rounds), MTP 47 / 51 (13) | verify 39.8 ms, draft 1.5 ms |
| FP4-Local, a 600-word essay | off | 97.6 | 2.23 | MTP 395 / 668 | verify 20.9 ms |
| | on | 93.3 | 2.22 | lookup 2 / 14 (2 rounds), MTP 435 / 727 | verify 22.0 ms |
| Qwen3.8-27B, the file rewrite | off | 135.4 | 4.62 | MTP 775 / 833 | verify 29.6 ms, draft 4.5 ms |
| | on | **212.7** | **7.27** | lookup 804 / 861 (123 rounds), MTP 49 / 52 (13) | verify 32.5 ms, draft 1.6 ms |
| Qwen3.8-27B, the essay | off / on | 77.0 / 77.0 | 2.25 | MTP 434 / 717, no lookup rounds | verify 26.8 ms |

Same greedy tokens on the copy task on both models. On the 27B (every weight in VRAM) the 8-row window costs
3 ms more than the 5-row one and the copy task runs 1.57x faster; on FP4-Local it costs 11 ms more (more
distinct experts per forward, so more CPU misses: 13.4 against 7.0 ms of host experts), which takes back part
of the 7.22 tokens per round, for 1.16x. Prose is untouched: the lookup rarely fires, and the agreement rule
keeps it out when it would be wrong.

## Prefix forking (`src/engine/park.cpp`, `serve/server.py`)

- The server snapshots the engine state at two positions per request: where the first turn after a leading
  system message begins (new) and where the last turn begins (as before). Requests carry `checkpoints: [a, b]`;
  `Generator::start` reads the prompt in segments and snapshots after each.
- `Engine::select_conversation` considers, beside the live slot and the parked conversations this request
  continues, every conversation holding a snapshot inside the shared prefix: a parked one, or an idle one on
  the GPU. The best is **forked**: its KV cells `[0, pos)` and the snapshot at `pos` are copied into the live
  slot (device to device for a slot, host to device for a parked entry) and the source keeps its own state; the
  live slot's own conversation is parked first when it is worth keeping (`park_min`). A fork must save at least
  64 tokens (`BNK_FORK_MIN`, 0 turns forking off); `park_stats.forks / fork_ms` count them.
- `serve`'s slot choice, when the engine can fork: a slot the prompt continues, else an empty one, else the
  least recently used, instead of rewinding a sibling's slot to the shared prefix.

Measured through the server (FP4-Local, 3 slots, greedy, a 3,590-token system prompt shared by conversations
A, B, E, F; X0-X2 share another; `scratchpad/fork_test.sh`): before, a sibling's first turn read the whole
prompt (no snapshot lay inside the shared prefix); now it starts from the shared system prompt.

| request | master | this branch |
|---|---|---|
| A: shared system prompt + question 1 (cold) | 2.9 s, reused 0 | 3.0 s, reused 0 |
| B: the same system prompt, question 2 | 5.5 s, reused 0 | **0.18 s, reused 3,594** (forked from A's slot in 16 ms) |
| A's follow-up (its last turn re-rendered) | 0.24 s, reused 3,609 (rewound in its slot) | 0.25 s, reused 3,594 |
| E, F: the system prompt again after three other conversations pushed A off the GPU | 3.1 s, reused 0 | **0.15 s, reused 3,594** (forked from A's parked copy in 21 ms) |
| X1, X2: two more conversations sharing X0's system prompt | 2.9-3.1 s, reused 0 | 0.11-0.12 s, reused 3,609 |

Every answer equals master's and the no-fork run's; `tools/park_check.py` (1 and 3 slots) and `tools/serve_check.py` pass. What the fork adds over the snapshot alone: the sibling's state is
copied, so A, B, E and F all stay resident (or parked whole) instead of A being rewound to the prefix for B
and parked / restored around every switch (0.17-0.3 s per park at this size, ~1 s at 60K).

Not done: forking in the dense engine (it needs parking first, item 9), and snapshots at every turn boundary
(bounded by 6 per slot today; the two positions cover a shared system prompt and a re-rendered last turn).
