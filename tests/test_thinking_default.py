#!/usr/bin/env python3
"""A config's chat_template_kwargs defaults (enable_thinking false, a reasoning_effort level) and what a request can
do over them: chat_template_kwargs, a top-level enable_thinking, reasoning_effort (pi sends that for a thinking level
on a "reasoning": true model; "none" for off). A request's level beats the config's, and every level reaches the
template in its own terms (Qwen3.8: low / medium / xhigh; DeepSeek-V4.1: low / high / max or 1-100).
    .venv/bin/python tests/test_thinking_default.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve import server  # noqa: E402


class Template:   # the end of Qwen3.8's template: an empty think block when thinking is off
    native_system = False

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        self.effort = kw.get("reasoning_effort")
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
# (template, config chat_template_kwargs, request extras, thinking expected, reasoning_effort the template gets)
EFFORT = [
    ("qwen", {"reasoning_effort": "low"}, {}, True, "low"),
    ("qwen", {"reasoning_effort": "low"}, {"reasoning_effort": "high"}, True, "xhigh"),
    ("qwen", {"reasoning_effort": "low"}, {"chat_template_kwargs": {"reasoning_effort": "medium"}}, True, "medium"),
    ("qwen", {"reasoning_effort": "low"}, {"reasoning_effort": "none"}, False, None),
    ("qwen", {"reasoning_effort": "high"}, {}, True, "xhigh"),
    ("qwen", {"reasoning_effort": "max"}, {}, True, "xhigh"),
    ("qwen", {"reasoning_effort": "minimal"}, {}, True, "low"),
    ("qwen", {"reasoning_effort": 30}, {}, True, "low"),
    ("qwen", {"reasoning_effort": "none"}, {}, False, None),
    ("qwen", {"reasoning_effort": "none"}, {"reasoning_effort": "low"}, True, "low"),
    ("qwen", {"reasoning_effort": "none"}, {"enable_thinking": True}, True, None),
    ("qwen", {"enable_thinking": False, "reasoning_effort": "low"}, {}, False, "low"),
    ("qwen", {"enable_thinking": False, "reasoning_effort": "low"}, {"reasoning_effort": "xhigh"}, True, "xhigh"),
    ("qwen", {}, {"reasoning_effort": "High"}, True, "xhigh"),
    ("DSML", {"reasoning_effort": "medium"}, {}, True, "62"),
    ("DSML", {"reasoning_effort": "low"}, {"reasoning_effort": "xhigh"}, True, "max"),
    ("DSML", {"reasoning_effort": 90}, {}, True, "90"),
    ("DSML", {"reasoning_effort": "high"}, {}, True, "high"),
    ("DSML", {}, {"reasoning_effort": "minimal"}, True, "25"),
]
fails = 0
for tmpl, cfg, extra, want, want_eff in EFFORT:
    S.tok.chat_template, S.config = tmpl, {"chat_template_kwargs": cfg}
    _, got = server.chat_prompt({**MSG, **extra})
    ok = got == want and S.template.effort == want_eff
    fails += not ok
    print(f"{tmpl} config {cfg} request {extra}: thinking {got}, effort {S.template.effort!r}"
          f"{'' if ok else f'  FAIL (want {want}, {want_eff!r})'}")
S.tok.chat_template = "qwen"
for default, extra, want in CASES:
    S.config = {} if default is None else {"chat_template_kwargs": {"enable_thinking": default}}
    _, got = server.chat_prompt({**MSG, **extra})
    ok = got == want
    fails += not ok
    print(f"default {default!s:5} request {extra}: thinking {got}{'' if ok else '  FAIL (want ' + str(want) + ')'}")
print("all passed" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
