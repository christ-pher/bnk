#!/usr/bin/env python3
# The DFlash2 drafter (e.g. incoai/Qwen3.8-27B-DFlash2, safetensors BF16) as a GGUF bnk loads (src/dense/dense_dflash.h).
# Matrices stay BF16 (bnk makes them Q8_0 at load, as it does the 27B's); fc is split into its five column blocks,
# one per target layer tap (bnk adds fc_i . h_i as each tap's residual is ready); norms and the convolutions' base
# kernels are F32; the selector's codebooks F16 (gathered by row).
#   python tools/dflash_gguf.py DRAFT_DIR OUT.gguf      (needs torch, safetensors and gguf-py: the converter venv)
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path.home() / ".cache/bnk/llama.cpp/gguf-py"))
import gguf  # noqa: E402

src, out = Path(sys.argv[1]), sys.argv[2]
cfg = json.load(open(src / "config.json"))
dc = cfg["dflash_config"]
E = cfg["hidden_size"]
taps = dc["target_layer_ids"]

w = gguf.GGUFWriter(out, "dflash2")
w.add_uint32("dflash2.hidden_size", E)
w.add_uint32("dflash2.intermediate_size", cfg["intermediate_size"])
w.add_uint32("dflash2.num_hidden_layers", cfg["num_hidden_layers"])
w.add_uint32("dflash2.num_attention_heads", cfg["num_attention_heads"])
w.add_uint32("dflash2.num_key_value_heads", cfg["num_key_value_heads"])
w.add_uint32("dflash2.head_dim", cfg["head_dim"])
w.add_float32("dflash2.rms_norm_eps", cfg["rms_norm_eps"])
w.add_float32("dflash2.rope_theta", cfg["rope_parameters"]["rope_theta"])
w.add_uint32("dflash2.sliding_window", cfg["sliding_window"])
w.add_uint32("dflash2.vocab_size", cfg["vocab_size"])
w.add_uint32("dflash2.block_size", dc["block_size"])
w.add_uint32("dflash2.mask_token_id", dc["mask_token_id"])
w.add_uint32("dflash2.conv_group_size", dc["conv_group_size"])
w.add_uint32("dflash2.conv_kernel_size", dc["conv_kernel_size"])
w.add_uint32("dflash2.selector_rank", dc["selector_rank"])
w.add_uint32("dflash2.selector_top_k", dc["selector_top_k"])
w.add_array("dflash2.target_layer_ids", taps)


def bf16(name, t):
    a = t.contiguous().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    w.add_tensor(name, a, raw_shape=a.shape, raw_dtype=gguf.GGMLQuantizationType.BF16)


st = safe_open(str(src / "model.safetensors"), "pt")
n = 0
for k in sorted(st.keys()):
    t = st.get_tensor(k)
    if k == "fc.weight":
        for i in range(len(taps)):
            bf16(f"fc.{i}.weight", t[:, i * E : (i + 1) * E])
    elif k.endswith("codebook"):
        w.add_tensor(k, t.float().numpy().astype(np.float16))
    elif t.dim() == 1 or k.endswith("base_kernel"):
        w.add_tensor(k, t.float().numpy())
    else:
        bf16(k, t)
    n += 1
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out}: {n} tensors (fc as {len(taps)} blocks)")
