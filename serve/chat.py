"""Chat template rendering and streaming output parsing (reasoning / content / tool calls)."""
from __future__ import annotations

import json
import re
import uuid

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


class ChatTemplate:
    def __init__(self, source: str):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def tojson(v, indent=None, **_):
            return json.dumps(v, ensure_ascii=False, indent=indent)

        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = tojson
        self.template = env.from_string(source)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs) -> str:
        msgs = []
        for m in messages:
            m = dict(m)
            # OpenAI tool calls carry JSON-string arguments; the template wants a mapping
            if m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    tc = json.loads(json.dumps(tc))
                    fn = tc.get("function", tc)
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                        except json.JSONDecodeError:
                            fn["arguments"] = {"input": fn["arguments"]}
                    calls.append(tc)
                m["tool_calls"] = calls
            msgs.append(m)
        kw = {k: v for k, v in kwargs.items() if v is not None}
        return self.template.render(messages=msgs, tools=tools or None, add_generation_prompt=add_generation_prompt,
                                    **kw)


_TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text: str):
    """-> (content without the calls, [OpenAI tool_call dicts])."""
    calls = []
    for m in _TOOL_RE.finditer(text):
        args = {}
        for p in _PARAM_RE.finditer(m.group(2)):
            raw = p.group(2)
            try:
                args[p.group(1).strip()] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                args[p.group(1).strip()] = raw
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": m.group(1).strip(), "arguments": json.dumps(args, ensure_ascii=False)}})
    content = _TOOL_RE.sub("", text).strip() if calls else text
    return content, calls


class OutputParser:
    """Streams model text into (kind, text) pieces: kind is 'reasoning' or 'content'. Tool-call markup is held
    back from the content stream and parsed at the end."""

    def __init__(self, thinking: bool):
        self.mode = "reasoning" if thinking else "content"
        self.buf = ""
        self.content_all = ""
        self.reasoning_all = ""
        self.in_tool = False

    def feed(self, text: str):
        self.buf += text
        out = []
        while self.buf:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if i < 0:
                    keep = _partial_suffix(self.buf, "</think>")
                    emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                    if emit:
                        out.append(("reasoning", emit))
                        self.reasoning_all += emit
                    break
                if i:
                    out.append(("reasoning", self.buf[:i]))
                    self.reasoning_all += self.buf[:i]
                self.buf = self.buf[i + len("</think>"):].lstrip("\n")
                self.mode = "content"
                continue
            # content: hold back anything that may start a tool call or a stray think tag
            if self.in_tool:
                self.content_all += self.buf
                self.buf = ""
                break
            i = self.buf.find("<tool_call>")
            if i >= 0:
                if i:
                    out.append(("content", self.buf[:i]))
                    self.content_all += self.buf[:i]
                self.content_all += self.buf[i:]
                self.buf = ""
                self.in_tool = True
                break
            if not self.content_all.strip() and not self.buf.strip():
                break   # only whitespace so far: hold it (leading blank lines are dropped below)
            if not self.content_all.strip():
                self.buf = self.buf.lstrip()
            if self.buf.startswith("<think>") and not self.content_all.strip():
                self.buf = self.buf.lstrip()[len("<think>"):]
                self.mode = "reasoning"
                continue
            keep = max(_partial_suffix(self.buf, "<tool_call>"), _partial_suffix(self.buf, "<think>"))
            emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
            if emit:
                out.append(("content", emit))
                self.content_all += emit
            break
        return out

    def finish(self):
        out = []
        if self.buf:
            kind = self.mode
            out.append((kind, self.buf))
            if kind == "reasoning":
                self.reasoning_all += self.buf
            else:
                self.content_all += self.buf
            self.buf = ""
        content, calls = parse_tool_calls(self.content_all)
        return out, content.strip() if calls else self.content_all, calls


def _partial_suffix(s: str, tag: str) -> int:
    """Length of the longest suffix of s that is a proper prefix of tag."""
    for n in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:n]):
            return n
    return 0
