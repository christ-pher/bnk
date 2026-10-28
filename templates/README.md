# Chat templates

A config's `"chat_template_file"` (a path from the repository root) replaces the chat template embedded in the GGUF.

| File | Used by | Source |
|---|---|---|
| `deepseek-v4.1.jinja` | ds-xs, ds-xs-abl | ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF `chat_template.jinja` at `a419d74c` (based on Unsloth's) |

`deepseek-v4.1.jinja` replaces the V4 template embedded in our pinned GGUF (`85caaf74`). The old one, even with
`dsml_v41_template`'s tag fix, left out the BOS token, rendered parallel tool results in arrival order instead of
call order and knew no image blocks. Against DeepSeek's reference encoder (`encoding/encoding.py` of
deepseek-ai/DeepSeek-V4.1-Flash-1017), rendered through the server's message handling, `tools/ds_template_check.py`
finds every case identical except one that only DeepSeek's app sends (a `latest_reminder` role, tools on a later
system message).
