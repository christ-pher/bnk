#!/usr/bin/env python3
# A DFlash2 drafter shipped as llama.cpp's "dflash" GGUF (e.g. orcarouter/OrcaCyber-Nano-1.0's
# *-DFlash2-Q8_0.gguf) rewritten as the "dflash2" GGUF bnk loads (src/dense/dense_dflash.h; tools/dflash_gguf.py
# writes the same layout from safetensors). Quantized matrices are copied as they are (bnk loads Q8_0 directly);
# fc is split into its five column blocks, one per tap; the selector's codebooks become F16 (gathered by row).
# llama.cpp stores the taps as layer ids + 1 (the layer's output); bnk takes the HF ids.
#   python tools/dflash_from_llamacpp.py IN.gguf OUT.gguf      (needs numpy and gguf-py)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / ".cache/bnk/llama.cpp/gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
if r.fields["general.architecture"].contents() != "dflash":
    sys.exit(f"{src}: not a llama.cpp dflash GGUF")
f = {k: v.contents() for k, v in r.fields.items() if not k.startswith("GGUF.")}
t = {x.name: x for x in r.tensors}
E = f["dflash.embedding_length"]
taps = [i - 1 for i in f["dflash.target_layers"]]
NL = f["dflash.block_count"]

w = gguf.GGUFWriter(out, "dflash2")
w.add_uint32("dflash2.hidden_size", E)
w.add_uint32("dflash2.intermediate_size", f["dflash.feed_forward_length"])
w.add_uint32("dflash2.num_hidden_layers", NL)
w.add_uint32("dflash2.num_attention_heads", f["dflash.attention.head_count"])
w.add_uint32("dflash2.num_key_value_heads", f["dflash.attention.head_count_kv"])
w.add_uint32("dflash2.head_dim", f["dflash.attention.key_length"])
w.add_float32("dflash2.rms_norm_eps", f["dflash.attention.layer_norm_rms_epsilon"])
w.add_float32("dflash2.rope_theta", f["dflash.rope.freq_base"])
w.add_uint32("dflash2.sliding_window", f["dflash.attention.sliding_window"])
w.add_uint32("dflash2.vocab_size", int(t["selector_predecessor.weight"].shape[1]))
w.add_uint32("dflash2.block_size", f["dflash.block_size"])
w.add_uint32("dflash2.mask_token_id", f["tokenizer.ggml.mask_token_id"])
w.add_uint32("dflash2.conv_group_size", f["dflash.conv_group_size"])
w.add_uint32("dflash2.conv_kernel_size", f["dflash.conv_kernel_size"])
w.add_uint32("dflash2.selector_rank", f["dflash.selector_rank"])
w.add_uint32("dflash2.selector_top_k", f["dflash.selector_top_k"])
w.add_array("dflash2.target_layer_ids", taps)


def copy(name, x):
    if x.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        w.add_tensor(name, np.asarray(x.data))
    else:
        w.add_tensor(name, np.asarray(x.data), raw_dtype=x.tensor_type)


# fc (E rows x taps*E columns): each tap's column block, cut on the quant blocks' byte boundary
fc = t["fc.weight"]
bs, ts = gguf.GGML_QUANT_SIZES[fc.tensor_type]
if E % bs:
    sys.exit("fc blocks do not split on a quant block boundary")
row = (E // bs) * ts
for i in range(len(taps)):
    w.add_tensor(f"fc.{i}.weight", np.ascontiguousarray(np.asarray(fc.data)[:, i * row : (i + 1) * row]),
                 raw_dtype=fc.tensor_type)
copy("hidden_norm.weight", t["enc.output_norm.weight"])
copy("norm.weight", t["output_norm.weight"])
copy("candidate_selector.hidden_projection.weight", t["selector_hidden.weight"])
for a, b in (("predecessor", "predecessor_codebook"), ("successor", "successor_codebook")):
    x = t[f"selector_{a}.weight"]
    w.add_tensor(f"candidate_selector.{b}", dequantize(x.data, x.tensor_type).astype(np.float16))

names = {
    "attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_conv_proj.weight": "attention_conv.kernel_projection.weight", "attn_conv_base": "attention_conv.base_kernel",
    "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight", "ffn_conv_base": "mlp_conv.base_kernel",
}
for l in range(NL):
    for a, b in names.items():
        copy(f"layers.{l}.{b}", t[f"blk.{l}.{a}"])
n = 3 + len(taps) + 2 + NL * len(names)
if n != len(r.tensors) + len(taps) - 1:
    sys.exit(f"{len(r.tensors)} tensors in, {n} out: a tensor was not mapped")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out}: {n} tensors (fc as {len(taps)} blocks, taps {taps})")
