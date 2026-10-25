#!/usr/bin/env python3
"""The tool-call parser types arguments by the request's tool schema: a string parameter stays text even when the
text is JSON (a file's content), structured parameters are parsed; without a schema JSON is tried first.
    .venv/bin/python tests/test_tool_calls.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.chat import OutputParser, dsml_v41_template, parse_tool_calls  # noqa: E402

TEXT = ("<tool_call>\n<function=write_file>\n<parameter=path>\na.json\n</parameter>\n<parameter=content>\n"
        '{"k": [1, 2]}\n</parameter>\n<parameter=count>\n3\n</parameter>\n<parameter=opts>\n{"force": true}\n'
        "</parameter>\n<parameter=note>\nnull\n</parameter>\n</function>\n</tool_call>")
PROPS = {"path": {"type": "string"}, "content": {"type": "string"}, "count": {"type": "integer"},
         "opts": {"type": "object"}, "note": {"type": "string"}}
OPENAI = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": PROPS}}}]
ANTHROPIC = [{"name": "write_file", "input_schema": {"type": "object", "properties": PROPS}}]


def args_of(tools):
    content, calls = parse_tool_calls(TEXT, tools)
    assert content == "" and len(calls) == 1 and calls[0]["function"]["name"] == "write_file"
    return json.loads(calls[0]["function"]["arguments"])


fails = 0
for label, tools in (("openai", OPENAI), ("anthropic", ANTHROPIC)):
    a = args_of(tools)
    want = {"path": "a.json", "content": '{"k": [1, 2]}', "count": 3, "opts": {"force": True}, "note": "null"}
    ok = a == want
    fails += not ok
    print(f"{label:10s} {'OK' if ok else 'FAIL: ' + repr(a)}")
a = args_of(None)
ok = a["content"] == {"k": [1, 2]} and a["count"] == 3 and a["note"] is None and a["path"] == "a.json"
fails += not ok
print(f"{'no schema':10s} {'OK (JSON first, as before)' if ok else 'FAIL: ' + repr(a)}")

# DeepSeek-V4.1: the GGUF's template spells the tags as V4 did; the model writes a space and a short word
V4_TEMPLATE = ("{%- set dsml_token = '｜DSML｜' -%}{{ '<' + dsml_token + 'tool_calls>\\n<' + dsml_token + 'invoke name=\"f\">\\n<'"
               " + dsml_token + 'parameter name=\"k\" string=\"true\">v</' + dsml_token + 'parameter>\\n</' + dsml_token"
               " + 'invoke>\\n</' + dsml_token + 'tool_calls>' }}")
fixed = dsml_v41_template(V4_TEMPLATE, "DeepSeek-V4.1-Flash-1017 GSQ-RCO IQ2_XS")
ok = ("'tool_calls" not in fixed and "'invoke" not in fixed and "'parameter" not in fixed
      and fixed.count("' calls") == 2 and fixed.count("' invoke") == 2 and fixed.count("' parameter") == 2)
fails += not ok
print(f"{'v4.1 tags':10s} {'OK' if ok else 'FAIL: ' + fixed}")
ok = dsml_v41_template(V4_TEMPLATE, "DeepSeek-V4-Flash-0731") == V4_TEMPLATE   # V4 itself keeps its spelling
fails += not ok
print(f"{'v4 kept':10s} {'OK' if ok else 'FAIL'}")
# the fixed template renders what encoding.py renders (checked against the model's GGUF when it is on this machine)
GGUF = sorted(Path("/root/.cache/huggingface/hub").glob("models--*DeepSeek-V4.1*/snapshots/*/*IQ2_XS-00001-of-*.gguf"))
if GGUF:
    from serve.chat import ChatTemplate
    from serve.tokenizer import Tokenizer
    tok = Tokenizer.from_gguf(str(GGUF[0]), cache_dir="/root/.cache/bnk")
    tpl = ChatTemplate(dsml_v41_template(tok.chat_template, tok.model_name))
    msgs = [{"role": "user", "content": "edit a.py"},
            {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [
                {"type": "function", "function": {"name": "edit", "arguments": '{"path": "a.py", "n": 2}'}}]},
            {"role": "tool", "content": "done"}]
    text = tpl.render(msgs, tools=OPENAI, add_generation_prompt=True, enable_thinking=True)
    want = ['<｜DSML｜ calls>\n<｜DSML｜ invoke name="edit">\n<｜DSML｜ parameter name="path" string="true">a.py</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="n" string="false">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>',
            'writing a "<｜DSML｜ calls>" block', '<｜User｜><tool_result>done</tool_result><｜Assistant｜><think>',
            '<｜Assistant｜><think>r</think>']
    missing = [w for w in want if w not in text]
    ok = not missing and "tool_calls>" not in text and "｜DSML｜invoke" not in text and "｜DSML｜parameter" not in text
    fails += not ok
    print(f"{'gguf tpl':10s} {'OK' if ok else 'FAIL: missing ' + repr(missing)}")
    # the model's own output in that spelling parses, and the parsed call re-renders identically
    out = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="edit">\n<｜DSML｜ parameter name="path" string="true">a.py</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    _, calls = parse_tool_calls(out, OPENAI)
    ok = len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}
    fails += not ok
    print(f"{'v4.1 parse':10s} {'OK' if ok else 'FAIL: ' + repr(calls)}")
else:
    print(f"{'gguf tpl':10s} skipped (no DeepSeek-V4.1 GGUF here)")

# a tool call the generation was cut off inside: the parser keeps the content before it and the markup apart
p = OutputParser(thinking=True, tools=OPENAI)
pieces = []
for chunk in ("<think>plan</think>", "Editing now.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write_file\">\n",
              "<｜DSML｜ parameter name=\"content\" string=\"true\">line 1\nline 2\nline"):
    pieces += p.feed(chunk)
rest, content, calls = p.finish()
streamed = "".join(t for k, t in pieces + rest if k == "content")
ok = (not calls and content.startswith("Editing now.") and "<｜DSML｜" in content
      and p.clean_content == "Editing now.\n\n" and p.partial_call.startswith("<｜DSML｜ calls>")
      and streamed == "Editing now.\n\n")
fails += not ok
print(f"{'cut call':10s} {'OK' if ok else 'FAIL: ' + repr((content, p.clean_content, p.partial_call, streamed))}")
p = OutputParser(thinking=False, tools=OPENAI)
p.feed("Done. " + out if GGUF else "Done. <｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>")
rest, content, calls = p.finish()
ok = len(calls) == 1 and not p.partial_call and content == "Done."
fails += not ok
print(f"{'whole call':10s} {'OK' if ok else 'FAIL: ' + repr((content, calls, p.partial_call))}")

print("FAILURES" if fails else "all passed")
sys.exit(1 if fails else 0)
