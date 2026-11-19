#!/usr/bin/env python3
"""A config's chat_template_kwargs default (enable_thinking false for the pi adapter) and what a request can do over
it: chat_template_kwargs, a top-level enable_thinking, reasoning_effort (pi sends that for a thinking level on a
"reasoning": true model; "none" for off).
    .venv/bin/python tests/test_thinking_default.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve import server  # noqa: E402


class Template:   # the end of Qwen3.8's template: an empty think block when thinking is off
    native_system = False

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        return "<|im_start|>assistant\n" + ("<think>\n" if kw.get("enable_thinking", True) else "<think>\n\n</think>\n\n")


S = server.S = SimpleNamespace()
S.template = Template()
S.tok = SimpleNamespace(chat_template="qwen", encode_chat=lambda text: [])
S.images = None
MSG = {"messages": [{"role": "user", "content": "fix the bug"}]}
CASES = [
    # (config default, request extras, thinking expected)
    (None, {}, True),
    (None, {"reasoning_effort": "none"}, False),
    (None, {"chat_template_kwargs": {"enable_thinking": False}}, False),
    (False, {}, False),
    (False, {"reasoning_effort": "high"}, True),
    (False, {"reasoning_effort": "medium"}, True),
    (False, {"reasoning_effort": "none"}, False),
    (False, {"enable_thinking": True}, True),
    (False, {"chat_template_kwargs": {"enable_thinking": True}}, True),
    (False, {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}, False),
    (True, {"chat_template_kwargs": {"enable_thinking": False}}, False),
]
fails = 0
for default, extra, want in CASES:
    S.config = {} if default is None else {"chat_template_kwargs": {"enable_thinking": default}}
    _, got = server.chat_prompt({**MSG, **extra})
    ok = got == want
    fails += not ok
    print(f"default {default!s:5} request {extra}: thinking {got}{'' if ok else '  FAIL (want ' + str(want) + ')'}")
print("all passed" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
