#!/usr/bin/env python3
"""The tool-call parser types arguments by the request's tool schema: a string parameter stays text even when the
text is JSON (a file's content), structured parameters are parsed; without a schema JSON is tried first.
    .venv/bin/python tests/test_tool_calls.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.chat import parse_tool_calls  # noqa: E402

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
print("FAILURES" if fails else "all passed")
sys.exit(1 if fails else 0)
