#!/usr/bin/env python3
"""The server's request loop (serve/server.py run()) against a fake engine: the answer's loop guard ends a repeating
answer and collapses its repeated tool call, a tool call cut off at max_tokens is dropped from the content, and a
whole call still parses. Needs the DeepSeek-V4.1 GGUF's tokenizer (skips without it).
    BNK_TOKS=0 .venv/bin/python tests/test_serve_run.py"""
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import serve.server as server  # noqa: E402
from serve.tokenizer import Tokenizer  # noqa: E402

GGUF = sorted(Path("/root/.cache/huggingface/hub").glob("models--*DeepSeek-V4.1*/snapshots/*/*IQ2_XS-00001-of-*.gguf"))
if not GGUF:
    print("skipped (no DeepSeek-V4.1 GGUF here)")
    sys.exit(0)
tok = Tokenizer.from_gguf(str(GGUF[0]), cache_dir="/root/.cache/bnk")
TOOLS = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}}}}}]
CALL = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="write_file">\n<｜DSML｜ parameter name="path" string="true">a.py</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="content" string="true">print(1)\n</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')


class FakeEngine:
    """Streams `ids` in chunks of 7 and ends with `reason`; a cancel ends it early (as the real engine does)."""

    def __init__(self, ids, reason="stop"):
        self.ids, self.reason, self.cancelled = ids, reason, False
        self.info = {"n_ctx": 1 << 20}

    def generate(self, prompt, params, on_start=None):
        self.cancelled = False
        yield {"type": "prefill", "_rid": "r1", "ms": 1, "reused": 0}
        for i in range(0, len(self.ids), 7):
            if self.cancelled:
                yield {"type": "done", "_rid": "r1", "reason": "cancel", "gen_ms": 1}
                return
            yield {"type": "tokens", "_rid": "r1", "ids": self.ids[i:i + 7]}
        yield {"type": "done", "_rid": "r1", "reason": self.reason, "gen_ms": 1}

    def cancel(self, rid):
        self.cancelled = True

    def waiters(self):
        return 0


class Quiet:
    def __init__(self):
        self.lines = []

    def line(self, s):
        self.lines.append(s)

    def _c(self, code, s):
        return s


class Convs:
    def begin(self, ids):
        return 1

    def status(self, *a):
        pass

    def produced(self, *a):
        pass

    def end(self, *a):
        pass


def serve_fake(text_ids, reason, thinking=True):
    S = SimpleNamespace(engine=FakeEngine(text_ids, reason), console=Quiet(), convs=Convs(), lock=threading.Lock(),
                        waiting=0, active=None, tok=tok, stop_ids=[1], im_start=None, recent=[],
                        telemetry=SimpleNamespace(bus=SimpleNamespace(publish=lambda *a: None),
                                                  prompt_read=lambda *a: None),
                        args=SimpleNamespace(slice=0, think_guard=True, content_guard=True))
    server.S = S
    pieces = []
    out = server.run([0, 5, 6], {"temperature": 1.0}, 32000, [], thinking, lambda k, t: pieces.append((k, t)),
                     tools=TOOLS)
    content, reasoning, calls, finish, usage, timings = out
    return content, calls, finish, timings, "".join(t for k, t in pieces if k == "content"), S.console.lines


fails = 0
# 1. the answer repeats one tool call verbatim: the guard ends it, the client gets the call once
ids = tok.encode("plan</think>") + tok.encode("Writing.\n\n" + "\n\n".join([CALL] * 200))
content, calls, finish, timings, streamed, lines = serve_fake(ids, "length")
ok = (timings.get("content_loop_guard") == 1 and finish == "tool_calls" and len(calls) == 1
      and json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py", "content": "print(1)\n"}
      and content == "Writing.")
fails += not ok
print(f"{'loop+calls':12s} {'OK' if ok else 'FAIL'}: finish {finish}, {len(calls)} call(s), {timings.get('content_loop_guard')}, "
      f"{lines[-1] if lines else ''}")

# 2. the answer repeats prose with no call: ended as length, content kept
ids = tok.encode("plan</think>") + tok.encode(" ".join(["The quick brown fox jumps over the lazy dog, again and again."] * 400))
content, calls, finish, timings, streamed, lines = serve_fake(ids, "length")
ok = timings.get("content_loop_guard") == 1 and finish == "length" and not calls and content.startswith("The quick")
fails += not ok
print(f"{'loop prose':12s} {'OK' if ok else 'FAIL'}: finish {finish}, {len(content)} chars")

# 3. a tool call cut off at max_tokens: content before it, markup dropped, flagged
ids = tok.encode("plan</think>") + tok.encode("Editing now.\n\n" + CALL[:CALL.index("print(1)") + 5])
content, calls, finish, timings, streamed, lines = serve_fake(ids, "length")
ok = (finish == "length" and not calls and content == "Editing now." and streamed == "Editing now.\n\n"
      and timings.get("truncated_tool_call", 0) > 0 and "content_loop_guard" not in timings)
fails += not ok
print(f"{'cut call':12s} {'OK' if ok else 'FAIL'}: finish {finish}, content {content!r}, streamed {streamed!r}, "
      f"timings {timings.get('truncated_tool_call')}")

# 4. a whole call then EOS: parsed as before, nothing flagged
ids = tok.encode("plan</think>") + tok.encode("Done.\n\n" + CALL)
content, calls, finish, timings, streamed, lines = serve_fake(ids, "stop")
ok = (finish == "tool_calls" and len(calls) == 1 and content == "Done." and streamed == "Done.\n\n"
      and "truncated_tool_call" not in timings and "content_loop_guard" not in timings)
fails += not ok
print(f"{'whole call':12s} {'OK' if ok else 'FAIL'}: finish {finish}, {len(calls)} call(s), content {content!r}")

# 5. no thinking, plain prose, stop: untouched
ids = tok.encode("Just an answer.")
content, calls, finish, timings, streamed, lines = serve_fake(ids, "stop", thinking=False)
ok = finish == "stop" and content == "Just an answer." and streamed == "Just an answer." and not calls
fails += not ok
print(f"{'plain':12s} {'OK' if ok else 'FAIL'}: finish {finish}, content {content!r}")

print("FAILURES" if fails else "all passed")
sys.exit(1 if fails else 0)
