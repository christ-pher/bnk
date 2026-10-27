# Abliterated DeepSeek-V4.1-Flash (`ds-xs-abl`)

[huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF)
removes refusals by editing one tensor a layer, `blk.N.attn_output_b.weight` (the second half of the attention output
projection, 8192 x 5120, Q5_K). The experts are untouched. It ships full Unsloth builds (UD-IQ1_S/IQ1_M) and patch files
for Strata and Maya. The Strata patch (`Strata-deepseek-v4.1-flash-IST-DAS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.gguf`,
1.16 GB, 40 tensors) is cut from ISTA's GSQ-RCO IQ2_XXS. It fits ds-xs as well: the 40 `attn_output_b` tensors of ISTA's
IQ2_XS and IQ2_XXS builds are byte-identical (all Q5_K, all in shard 4).

## What the patch changes (measured 2026-10-27)

| layers | bytes changed | ‖ΔW‖/‖W‖ | ΔW's spectrum |
|---|---|---|---|
| 0-10, 29-39 | 29-33% | 0.040-0.042 | flat (singular values 0.144 ...): requantization noise only |
| 11-28 | 53-57% | 0.079-0.088 | one direction (top singular value 9.61, then 0.186, 0.178 ...) + the same noise |

The ablation (a rank-1 projection out of the refusal direction) is in layers 11-28 only, as Huihui's README says for the
Unsloth builds. The Strata README patches layers 0-40, which only adds a second Q5_K rounding to 22 layers. We take
11-28 (the Maya README's range).

## Building the model directory

```sh
mkdir -p ~/.cache/bnk/huihui
.venv/bin/hf download huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF \
    Strata-deepseek-v4.1-flash-IST-DAS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.gguf --local-dir ~/.cache/bnk/huihui
.venv/bin/python tools/tensor_patch.py \
    --model "$(./models.sh resolve ds-xs | cut -f2)" \
    --patch ~/.cache/bnk/huihui/Strata-deepseek-v4.1-flash-IST-DAS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS.gguf \
    --out ~/.cache/bnk/huihui/IQ2_XS-abliterated --layers 11-28
```

Shards 1-3 and the Engram tables become symlinks into ds-xs's snapshot. Shard 4 is copied (36.8 GB) and its 18 tensors
are overwritten in place: same type and size, so the GGUF layout is unchanged. The result is read back and checked. ds-xs
has to stay downloaded.

## Running

`./run.sh ds-xs-abl` runs with ds-xs's engine settings, sampling and DSpark drafter. The routing counts (expert placement)
are kept per config. The experts are the same, so ds-xs's file can seed them:
`cp ~/.cache/bnk/counts-DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS.bnkc ~/.cache/bnk/counts-Huihui-DeepSeek-V4.1-Flash-1017-abliterated-IQ2_XS.bnkc`.
