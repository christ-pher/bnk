# oh-my-pi (omp) setup for bnk

Research notes, 2026-10-07. omp runs on the client machine, so its config was not read directly. The findings come
from the bnk server's live stats (`/api/stats`) and omp's own docs.

## What the server shows

- **Compaction runs, but barely shrinks anything.** The DFIR (CyLR) session grew to 181K tokens. omp then compacted
  it, and the new conversation still starts at 88K tokens of "serialized record" and grew back to 115K. For a 3-bit
  model that is most of the problem: it works at 90-180K context almost all the time.
- **omp only compacts at about 222K.** By default it waits until the context window minus
  `max(16384, 15%)` (262K - 39K). The model's quality drops long before that.
- **omp's default compaction method is risky with our server.** `methodOrder` defaults to
  `[remote, snapcompact, handoff, shake, soft]`. `snapcompact` turns old history into PNG images when the model is
  declared able to read images. bnk's server keeps only the text parts of a message, so imaged history would be
  dropped without any error. Whether this happened depends on what `models.yml` declares.
- **Too many parallel agents.** omp's defaults are `task.maxConcurrency: 32`, `task.maxRecursionDepth: 2`
  (subagents can spawn their own), no limit on requests in flight, and `compaction.asyncEnabled: true` (a
  speculative background summary request). Together that is more than bnk's 2 slots can hold.
- **Correction to an earlier diagnosis:** omp's docs say subagents do **not** inherit the parent's history, so the
  note that "each subagent starts from the parent's context" was probably wrong. The third big context may have been
  the background summary request. Not confirmed.

## Recommended omp config

`~/.omp/agent/config.yml`. Replace `bnk/<model>` with the provider and model id used in `models.yml`:

```yaml
modelRoles:
  default: bnk/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S:medium
  plan:    bnk/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S:high
  task:    bnk/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S:low
  smol:    bnk/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S:low
  commit:  bnk/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S:low

task:
  maxConcurrency: 1        # main agent + 1 subagent = our 2 slots
  maxRecursionDepth: 1     # subagents can't spawn more subagents

providers:
  maxInFlightRequests:
    bnk: 2                 # hard limit of 2 requests at once
  streamFirstEventTimeoutSeconds: 900   # long prompt reads shouldn't time out
  streamIdleTimeoutSeconds: 900

provider:
  appendOnlyContext: "on"  # history is only appended, never rewritten, so our prefix reuse keeps working

compaction:
  thresholdTokens: 96000   # compact at ~96K instead of ~222K (also applies to subagents)
  methodOrder: [handoff, soft, shake]   # no snapcompact or remote
  asyncEnabled: false      # no background summary request competing for slots
  keepRecentTokens: 20000

tools:
  artifactSpillThreshold: 10   # tool output over 10 KB goes to a file (default 50 KB)
  artifactHeadBytes: 4
  artifactTailBytes: 6
```

In `models.yml`, set the model's `input: [text]`, `contextWindow: 262144` and `maxTokens: 32768`.

Notes on these choices:

- **`handoff` first:** the model writes its own handoff note from the conversation it already has. That reuses
  bnk's cached prefix, so it only costs decode time, not a re-read of the prompt.
- **`:low` for subagents:** bnk's server passes the effort level to the chat template as `reasoning_effort`, so it
  does control how much the model thinks.
- **Similar setup elsewhere:** someone running omp on a local Qwen3.8 model made almost the same changes (a fixed
  compaction trigger, append-only context, smaller tool output, capped concurrency). Their cache hit rate went from
  55% to 95% and their slowest turns from 390 s to 77 s.
- Per-agent compaction triggers are possible with `task.agentCompactionThresholdOverrides` (agent name -> token
  count or `"N%"`).

## What the research says

- **Hugging Face, "Don't Train the Model, Evolve the Harness":** they kept a model (DeepSeek-V4-Pro) frozen and only
  rewrote the code around it. Its whole-task score went from 0% to 5%, its criterion pass rate from 63% to 80%, and
  it roughly matched Sonnet 4.6 at about 7x lower cost. Most of the gain came from fixing mechanical failures (files
  saved in the wrong place, broken tool-call JSON, loops), not from better prompts. Code fixes carried over to other
  models; prompt tweaks did not.
- **Hugging Face, "Is it agentic enough?":** extra docs in context helped big models but hurt small ones: Qwen3-14B
  dropped from 67% to 43%. Test harness changes against weaker models, and measure turns and tokens, not only
  correctness.
- **"An Empirical Study of Harness Design for Coding Agents":** context management mattered most, adding +35.7
  points at a 32K budget (+2.7 at 128K). Rule-based trimming followed by an LLM summary was the cheapest approach.
  For smaller models (around 30B), a fixed set of tools beat bash-only by 15%, and planning helped (+11.6%, at more
  turns). For large models, bash-only and no planning were cheaper.
- **Takeaway for bnk:** our 3-bit MoE model belongs in the "weaker model" group. Use a small tool set, compact early
  and run few parallel agents.

## Other harnesses worth trying

- **Plain pi** (the project omp is a fork of): much smaller system prompt and tool set. Hugging Face used it for their
  open-model tests. Best A/B test against omp.
- **Qwen Code:** built for Qwen models' tool-call format.

Plan: keep omp with the config above first. If quality is still poor, run the same task in plain pi.

## Possible bnk fix

`/v1/models` does not report a context length, so harnesses have to guess or rely on manual config. Adding it
would let omp pick up the window automatically.

## Sources

- [Don't Train the Model, Evolve the Harness (HF)](https://joelniklaus-harness-optimization.hf.space/)
- [Is it agentic enough? (HF blog)](https://huggingface.co/blog/is-it-agentic-enough)
- [An Empirical Study of Harness Design for Coding Agents](https://huggingface.co/papers/2609.20804)
- [oh-my-pi compaction docs](https://github.com/can1357/oh-my-pi/blob/main/docs/compaction.md)
- [oh-my-pi settings docs](https://github.com/can1357/oh-my-pi/blob/main/docs/settings.md)
- [oh-my-pi task tool docs](https://github.com/can1357/oh-my-pi/blob/main/docs/tools/task.md)
- [Tuning a Local Coding Agent: Oh My Pi and Qwen3.8-27B](https://doug.sh/posts/tuning-a-local-coding-agent-oh-my-pi/)
