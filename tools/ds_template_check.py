#!/usr/bin/env python3
"""Render DeepSeek-V4.1 conversations with a chat template through the server's message handling (serve/chat.py) and
compare the prompts with DeepSeek's reference encoder (encoding/encoding.py of the base repository, fetched once into
~/.cache/bnk/ds-encoding): the five test cases shipped with it, agent turns with parallel tool calls and results out of
order, mid-conversation system messages, reasoning efforts, images in user and tool messages.

    tools/ds_template_check.py [TEMPLATE] [-v]      (default: templates/deepseek-v4.1.jinja; -v prints each difference)
"""
import copy, json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve.chat import ChatTemplate, dsml_v41_template

REPO, REV = "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170"
ENC = Path("~/.cache/bnk/ds-encoding").expanduser()
for f in ["encoding/encoding.py"] + [f"encoding/tests/test_input_{i}.json" for i in range(1, 6)]:
    hf_hub_download(REPO, f, revision=REV, local_dir=str(ENC))
sys.path.insert(0, str(ENC / "encoding"))
import encoding as enc

args = [a for a in sys.argv[1:] if a != "-v"]
path = Path(args[0]) if args else ROOT / "templates/deepseek-v4.1.jinja"
tmpl = ChatTemplate(dsml_v41_template(path.read_text(), "DeepSeek-V4.1"))

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                         "required": ["path"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Run a command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


AGENT = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "What is in main.py?"},
    {"role": "assistant", "reasoning_content": "Read it.", "content": "",
     "tool_calls": [call(1, "read_file", {"path": "main.py", "limit": 50})]},
    {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"},
    {"role": "assistant", "reasoning_content": "Simple.", "content": "It prints hi."},
    {"role": "user", "content": "Run it and list the folder."},
    {"role": "assistant", "reasoning_content": "Two calls.", "content": "Running both.",
     "tool_calls": [call(2, "bash", {"command": "python main.py"}), call(3, "bash", {"command": "ls"})]},
    {"role": "tool", "tool_call_id": "call_3", "content": "main.py"},   # out of call order
    {"role": "tool", "tool_call_id": "call_2", "content": "hi"},
]
CASES = []
for i in range(1, 6):
    for c in enc.load_cases(str(ENC / f"encoding/tests/test_input_{i}.json")):
        CASES.append((f"ds-test-{i}", c["messages"], c.get("thinking_mode") or "thinking", c.get("reasoning_effort")))
for think in ("thinking", "chat"):
    CASES += [
        (f"agent/{think}", AGENT, think, None),
        (f"agent-gen/{think}", AGENT[:4], think, None),
        (f"plain/{think}", [{"role": "user", "content": "hello"}], think, None),
        (f"sys-mid/{think}", [{"role": "system", "content": "S"}, {"role": "user", "content": "a"},
                              {"role": "assistant", "content": "b"}, {"role": "system", "content": "reminder"},
                              {"role": "user", "content": "c"}], think, None),
        (f"multiturn/{think}", [{"role": "user", "content": "a"}, {"role": "assistant", "reasoning_content": "r1", "content": "b"},
                                {"role": "user", "content": "c"}], think, None),
    ]
for eff in ("low", "high", "max", 30, "62"):
    CASES.append((f"effort={eff}", [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], "thinking", eff))
CASES.append(("image", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                                      {"type": "text", "text": "what is this"}]}], "thinking", None))
CASES.append(("image-in-tool", [{"role": "system", "content": "S"}, {"role": "user", "content": "screenshot"},
                                {"role": "assistant", "content": "", "tool_calls": [call(1, "bash", {"command": "shot"})]},
                                {"role": "tool", "tool_call_id": "call_1", "content": [
                                    {"type": "text", "text": "here"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
              "thinking", None))


def reference(msgs, think, eff, tools):
    m = copy.deepcopy(msgs)
    if tools and not any(x.get("tools") for x in m):
        m[0]["tools"] = tools
    return enc.encode_messages(m, thinking_mode=think, reasoning_effort=eff)


def split_tools(msgs):
    """encoding.py takes tools on a message; templates take a tools argument."""
    m = copy.deepcopy(msgs); tools = None
    for x in m:
        if "tools" in x:
            tools = x.pop("tools")
    return m, tools


def render(t, msgs, think, eff, tools, server):
    kw = {"enable_thinking": think == "thinking", "thinking": think == "thinking"}
    if eff is not None:
        kw["reasoning_effort"] = eff
    if server:
        return t.render(msgs, tools=tools, **kw)
    return t.template.render(messages=msgs, tools=tools, add_generation_prompt=True, **kw)


def first_diff(a, b):
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return f"@{i}: ref {a[max(0, i - 60):i + 80]!r}\n         got {b[max(0, i - 60):i + 80]!r}"


res = []
for name, msgs, think, eff in CASES:
    m, tools = split_tools(msgs)
    use_tools = tools or (TOOLS if name.startswith("agent") or name == "image-in-tool" else None)
    try:
        ref = reference(msgs, think, eff, use_tools)
    except Exception as e:   # e.g. a numeric string effort, which the server sends and the reference refuses
        print(f"{name:22s} (no reference: {e})"); continue
    try:
        got = render(tmpl, copy.deepcopy(m), think, eff, use_tools, True)
    except Exception as e:
        got = f"<error {type(e).__name__}: {e}>"
    res.append(got == ref)
    print(f"{name:22s} {'same' if res[-1] else 'DIFFERENT'}")
    if not res[-1] and "-v" in sys.argv:
        print(f"   {first_diff(ref, got)}")
print(f"{path.name}: {sum(res)}/{len(res)} prompts identical to encoding.py")
