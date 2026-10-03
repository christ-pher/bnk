"""bnk HTTP server: OpenAI- and Anthropic-compatible APIs, live stats, and the web UI.

    python -m serve.server --model /path/model-00001-of-0000N.gguf [--mtp mtp.gguf] [--port 8080] [engine args...]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .chat import ChatTemplate, OutputParser
from .console import LEVELS, Console
from .engine import Engine, EngineError
from .loop_guard import WRAP_UP, ThinkingLoopGuard, at_clean_boundary
from .telemetry import Telemetry
from .tokenizer import StreamDecoder, Tokenizer

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web" / "dist"      # the built dashboard (serve/web: npm run build)
REQ = threading.local()           # per-request context: which API it came through


class State:
    def __init__(self, args):
        self.args = args
        self.tok = Tokenizer.from_gguf(args.model, cache_dir=str(Path(args.cache_dir).expanduser()))
        self.template = ChatTemplate(self.tok.chat_template)
        self.model_id = args.model_id or Path(args.model).name.split("-0000")[0].replace(".gguf", "")
        self.recent = collections.deque(maxlen=200)   # finished requests, for the monitor
        self.started = time.time()
        self.active = None                               # the request being served
        self.waiting = 0
        self.lock = threading.Lock()
        # stop tokens: EOS plus the chat turn end
        self.stop_ids = sorted({t for t in [self.tok.ids.get("eos_token_id"), self.tok.id_of("<|im_end|>"),
                                            self.tok.id_of("<|endoftext|>")] if t is not None})
        samp = self.tok.sampling
        self.defaults = {"temperature": float(samp.get("temp", 1.0)), "top_k": int(samp.get("top_k", 20)),
                         "top_p": float(samp.get("top_p", 0.95))}
        eargs = ["--model", args.model, "--ctx", str(args.ctx)]
        if args.mtp:
            eargs += ["--mtp", args.mtp, "--draft", str(args.draft)]
            if args.draft_vocab:
                eargs += ["--draft-vocab", args.draft_vocab]
        if args.profile:
            eargs += ["--profile", args.profile]
        if args.counts:
            eargs += ["--counts", args.counts]
        eargs += args.engine_args
        self.telemetry = Telemetry()
        self.console = Console(self.telemetry.bus, args.log_level)
        self.engine = Engine(args.engine, eargs, log_path=args.log, on_telemetry=self.telemetry.on_engine,
                             on_log=lambda line: self.telemetry.bus.publish("log", {"t": time.time(), "line": line}))


S: State | None = None


def now() -> int:
    return int(time.time())


# ------------------------------------------------------------------------------------------- generation
def sampling_from(body: dict) -> dict:
    d = S.defaults
    p = {
        "temperature": float(body.get("temperature", d["temperature"])),
        "top_k": int(body.get("top_k", d["top_k"])),
        "top_p": float(body.get("top_p", d["top_p"])),
        "min_p": float(body.get("min_p", 0.0)),
        "presence_penalty": float(body.get("presence_penalty", 0.0)),
        "repetition_penalty": float(body.get("repetition_penalty", 1.0)),
    }
    if body.get("seed") is not None:
        p["seed"] = int(body["seed"])
    return p


def run(prompt_ids: list[int], params: dict, max_tokens: int, stops: list[str], thinking: bool, emit):
    """Drives one engine request. `emit(kind, text)` receives reasoning/content pieces as they decode.
    Returns (content, reasoning, tool_calls, finish_reason, usage, timings).

    Thinking-loop guard: while the model reasons, its tokens go through a ThinkingLoopGuard. When it reports a
    loop, generation runs on to the end of the line or sentence, the engine request is cancelled, and a second one
    continues from everything generated so far plus a short wrap-up that closes the reasoning, so the model writes
    its answer in the output budget left. The engine reuses its state (the new prompt extends what it has seen),
    so the hand-over costs one short prefill however long the conversation."""
    dec = StreamDecoder(S.tok)
    parser = OutputParser(thinking)
    guard = ThinkingLoopGuard() if (thinking and S.args.think_guard) else None
    force_at = int(os.environ.get("BNK_THINK_GUARD_TEST", "0"))   # testing: report a loop after N thinking tokens
    text_so_far = ""
    reasoning_tail = ""
    finish = "length"
    done_msg = {}
    rid = None
    t0 = time.time()
    counted = True
    generated: list[int] = []
    interventions = 0
    stopped = False

    def stream(ids: list[int], allow_guard: bool) -> str:
        """One engine request; returns 'stop' (a stop string), 'loop' (the guard), or the engine's reason."""
        nonlocal rid, counted, done_msg, text_so_far, stopped, reasoning_tail
        gen = S.engine.generate(ids, {**params, "max_tokens": max_tokens - len(generated), "stop_ids": S.stop_ids})
        wrap_pending, wrap_deadline, outcome = False, 0, None
        for ev in gen:
            rid = ev.get("_rid")
            t = ev.get("type")
            if t == "prefill":
                with S.lock:
                    if counted:
                        S.waiting -= 1
                        counted = False
                    if S.active is None:
                        S.active = {"id": rid, "started": t0, "prompt_tokens": ev.get("prompt_tokens"),
                                    "reused": ev.get("reused"), "prefill_ms": ev.get("ms"), "gen_tokens": 0}
            elif t == "tokens":
                if stopped:
                    continue
                generated.extend(ev["ids"])
                piece = dec.feed(ev["ids"])
                with S.lock:
                    if S.active:
                        S.active["gen_tokens"] += len(ev["ids"])
                if stops:
                    text_so_far += piece
                    hit = min((text_so_far.find(x) for x in stops if x and x in text_so_far), default=-1)
                    if hit >= 0:
                        cut = len(text_so_far) - hit
                        piece = piece[:max(0, len(piece) - cut)]
                        stopped = True
                        outcome = "stop"
                        S.engine.cancel(rid)
                in_thinking = parser.mode == "reasoning"
                for kind, txt in parser.feed(piece):
                    emit(kind, txt)
                if allow_guard and guard and in_thinking and parser.mode == "reasoning" and outcome is None:
                    reasoning_tail = (reasoning_tail + piece)[-200:]
                    if not wrap_pending and (guard.push(ev["ids"]) or (force_at and guard.count >= force_at)):
                        wrap_pending, wrap_deadline = True, len(generated) + 48
                    if wrap_pending and (at_clean_boundary(reasoning_tail) or len(generated) >= wrap_deadline):
                        outcome = "loop"
                        S.engine.cancel(rid)
            elif t == "done":
                done_msg = ev
                if outcome is None:
                    outcome = ev.get("reason", "stop")
            elif t == "error":
                raise EngineError(ev.get("message", "engine error"))
        return outcome

    with S.lock:
        S.waiting += 1
    try:
        outcome = stream(prompt_ids, True)
        if outcome == "loop":
            interventions += 1
            room = max_tokens - len(generated)
            S.console.line(S.console._c("33", "↻ ") + f"thinking loop after {guard.count:,} thinking tokens "
                           f"({guard.last_ratio:.0%} repeated): closing the reasoning, {room:,} tokens left to answer")
            S.telemetry.bus.publish("log", {"t": time.time(), "line": f"thinking loop guard: intervened after "
                                                                       f"{guard.count} thinking tokens"})
            for kind, txt in parser.feed(dec.flush() + WRAP_UP):
                emit(kind, txt)
            if room > 0:
                outcome = stream(prompt_ids + generated + S.tok.encode(WRAP_UP), False)
            else:
                outcome = "length"
        if outcome == "stop" and stopped:
            finish = "stop"
        else:
            finish = {"stop": "stop", "length": "length", "cancel": "stop", "context": "length"}.get(outcome, "stop")
    except BaseException:
        if rid:
            S.engine.cancel(rid)
        raise
    finally:
        with S.lock:
            if counted:
                S.waiting -= 1
            S.active = None
    tail = dec.flush()
    for kind, txt in parser.feed(tail):
        emit(kind, txt)
    rest, content, calls = parser.finish()
    for kind, txt in rest:
        emit(kind, txt)
    if calls:
        finish = "tool_calls"
    usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated),
             "total_tokens": len(prompt_ids) + len(generated)}
    timings = {k: done_msg.get(k) for k in ("prefill_ms", "prefill_tps", "gen_ms", "tps", "reused", "rounds",
                                             "tokens_per_round", "accepted", "drafted", "expert_miss_rate",
                                             "verify_ms", "draft_ms", "cpu_expert_ms")}
    if interventions:
        timings["thinking_loop_guard"] = interventions
    rec = {"id": rid, "time": time.time(), "api": getattr(REQ, "api", ""), "prompt_tokens": len(prompt_ids),
           "gen_tokens": usage["completion_tokens"], "finish": finish, "temperature": params.get("temperature"),
           "max_tokens": max_tokens, "thinking": thinking, "loop_guard": interventions, **timings}
    S.recent.append(rec)
    S.telemetry.bus.publish("request", rec)
    return content, parser.reasoning_all, calls, finish, usage, timings


def chat_prompt(body: dict) -> tuple[list[int], bool]:
    kwargs = dict(body.get("chat_template_kwargs") or {})
    thinking = kwargs.get("enable_thinking", body.get("enable_thinking", True))
    if body.get("reasoning_effort") and "reasoning_effort" not in kwargs:
        eff = body["reasoning_effort"]
        if eff == "none":
            thinking = False
        else:
            kwargs["reasoning_effort"] = {"high": "xhigh", "minimal": "low"}.get(eff, eff)
    kwargs["enable_thinking"] = bool(thinking)
    text = S.template.render(body.get("messages", []), tools=body.get("tools"), add_generation_prompt=True, **kwargs)
    return S.tok.encode(text), bool(thinking)


# ------------------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "bnk"

    def log_message(self, fmt, *a):
        if S.args.verbose or S.args.log_level == "debug":
            S.console.line("http: %s %s" % (self.address_string(), fmt % a))

    # ---- helpers
    def _auth(self) -> bool:
        key = S.args.api_key
        if not key:
            return True
        h = self.headers.get("Authorization", "") or ""
        if h == f"Bearer {key}" or self.headers.get("x-api-key") == key:
            return True
        self._json(401, {"error": {"message": "invalid API key", "type": "auth"}})
        return False

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj, event: str | None = None):
        s = (f"event: {event}\n" if event else "") + "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
        self.wfile.write(s.encode())
        self.wfile.flush()

    # ---- routes
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": S.model_id, "object": "model", "created": now(),
                                                                  "owned_by": "bnk"}]})
        if path == "/api/stats":
            return self._json(200, self._overview())
        if path == "/api/stream":
            return self._stream()
        if path == "/api/layers":
            q = dict(x.split("=", 1) for x in self.path.partition("?")[2].split("&") if "=" in x)
            w = q.get("window", "life")
            return self._json(200, S.telemetry.layer_window(None if w == "life" else float(w)))
        if path == "/health":
            return self._json(200 if S.engine.alive() else 503, {"status": "ok" if S.engine.alive() else "down"})
        if path.startswith("/v1/") or path.startswith("/api/"):
            return self._json(404, {"error": {"message": "not found"}})
        # the dashboard: files of the build, and index.html for anything else (client-side routes)
        return self._static(path.lstrip("/") or "index.html")

    def _overview(self) -> dict:
        with S.lock:
            waiting = S.waiting
        info = S.engine.info
        return {"model": S.model_id, "uptime": time.time() - S.started, "started": S.started,
                "alive": S.engine.alive(), "waiting": waiting, "defaults": S.defaults,
                "engine": {"n_ctx": info.get("n_ctx"), "n_vocab": info.get("n_vocab"), "mtp": info.get("mtp"),
                           "name": info.get("model"), "args": S.engine.args},
                "gpu": S.telemetry.gpu.static, "live": S.telemetry.live(),
                "history": list(S.telemetry.history), "recent": list(S.recent),
                "log": list(S.engine.log_tail)[-200:]}

    def _stream(self):
        """Server-sent events: `init` (everything), then `live` (~4/s), `sample` (1/s), `request`, `log`."""
        q = S.telemetry.bus.subscribe()
        try:
            self._sse_start()
            self._sse(self._overview(), "init")
            last = time.time()
            while True:
                try:
                    event, data = q.get(timeout=10)
                    self._sse(data, event)
                except queue.Empty:
                    pass
                if time.time() - last > 10:   # keep proxies and the browser from timing the stream out
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            S.telemetry.bus.unsubscribe(q)

    def _static(self, name):
        root = WEB.resolve()
        p = (WEB / name).resolve()
        if not str(p).startswith(str(root)) or not p.is_file():
            p = root / "index.html"
            if not p.is_file():
                return self._json(404, {"error": {"message": "dashboard not built: cd serve/web && npm run build"}})
        data = p.read_bytes()
        ctype = {".html": "text/html; charset=utf-8", ".js": "application/javascript", ".css": "text/css",
                 ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".woff2": "font/woff2",
                 ".json": "application/json"}.get(p.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # hashed build assets never change; the page itself is always revalidated
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if "/assets/" in str(p) else "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = self.path.split("?")[0]
        REQ.api = {"/v1/chat/completions": "openai", "/v1/completions": "completions",
                   "/v1/messages": "anthropic"}.get(path, path)
        if self.headers.get("X-Bnk-Client") == "dashboard":
            REQ.api = "dashboard"
        if not self._auth():
            return
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._json(400, {"error": {"message": "invalid JSON"}})
        try:
            if path == "/v1/chat/completions":
                return self._chat(body)
            if path == "/v1/completions":
                return self._completions(body)
            if path == "/v1/messages":
                return self._anthropic(body)
            if path == "/api/reset":
                return self._json(200, {"ok": True})
            self._json(404, {"error": {"message": "not found"}})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._json(500, {"error": {"message": str(e), "type": type(e).__name__}})
            except Exception:
                pass

    # ---- OpenAI chat
    def _chat(self, body):
        prompt, thinking = chat_prompt(body)
        if len(prompt) >= S.engine.info.get("n_ctx", 1 << 30):
            return self._json(400, {"error": {"message": f"prompt of {len(prompt)} tokens exceeds the context"}})
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or S.args.max_tokens)
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        params = sampling_from(body)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = now()
        model = S.model_id
        if not body.get("stream"):
            content, reasoning, calls, finish, usage, timings = run(prompt, params, max_tokens, stops, thinking,
                                                                    lambda k, t: None)
            msg = {"role": "assistant", "content": content if (content or not calls) else None}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                    "usage": usage, "timings": timings})
        self._sse_start()
        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        self._sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                        "finish_reason": None}]})

        def emit(kind, txt):
            if not txt:
                return
            key = "reasoning_content" if kind == "reasoning" else "content"
            self._sse({**base, "choices": [{"index": 0, "delta": {key: txt}, "finish_reason": None}]})

        content, reasoning, calls, finish, usage, timings = run(prompt, params, max_tokens, stops, thinking, emit)
        if calls:
            self._sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": i, **c} for i, c in enumerate(calls)]}, "finish_reason": None}]})
        self._sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "timings": timings})
        if (body.get("stream_options") or {}).get("include_usage"):
            self._sse({**base, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- OpenAI completions
    def _completions(self, body):
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt and isinstance(prompt[0], str) else ""
        ids = S.tok.encode(prompt)
        max_tokens = int(body.get("max_tokens") or 256)
        stops = body.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        params = sampling_from(body)
        cid = "cmpl-" + uuid.uuid4().hex[:24]
        if not body.get("stream"):
            content, _, _, finish, usage, timings = run(ids, params, max_tokens, stops, False, lambda k, t: None)
            return self._json(200, {"id": cid, "object": "text_completion", "created": now(), "model": S.model_id,
                                    "choices": [{"index": 0, "text": content, "finish_reason": finish}],
                                    "usage": usage, "timings": timings})
        self._sse_start()

        def emit(kind, txt):
            if txt:
                self._sse({"id": cid, "object": "text_completion", "created": now(), "model": S.model_id,
                           "choices": [{"index": 0, "text": txt, "finish_reason": None}]})

        _, _, _, finish, usage, _ = run(ids, params, max_tokens, stops, False, emit)
        self._sse({"id": cid, "object": "text_completion", "created": now(), "model": S.model_id,
                   "choices": [{"index": 0, "text": "", "finish_reason": finish}], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- Anthropic messages
    def _anthropic(self, body):
        msgs = []
        system = body.get("system")
        if system:
            msgs.append({"role": "system", "content": system if isinstance(system, str) else
                         "\n".join(b.get("text", "") for b in system if b.get("type") == "text")})
        for m in body.get("messages", []):
            c = m.get("content")
            if isinstance(c, str):
                msgs.append({"role": m["role"], "content": c})
                continue
            text, reasoning, calls, results = [], "", [], []
            for b in c or []:
                t = b.get("type")
                if t == "text":
                    text.append(b.get("text", ""))
                elif t == "thinking":
                    reasoning += b.get("thinking", "")
                elif t == "tool_use":
                    calls.append({"id": b.get("id"), "type": "function",
                                  "function": {"name": b.get("name"), "arguments": b.get("input") or {}}})
                elif t == "tool_result":
                    rc = b.get("content")
                    if isinstance(rc, list):
                        rc = "\n".join(x.get("text", "") for x in rc if x.get("type") == "text")
                    results.append(rc or "")
            if results:
                for r in results:
                    msgs.append({"role": "tool", "content": r})
                if text:
                    msgs.append({"role": "user", "content": "\n".join(text)})
                continue
            mm = {"role": m["role"], "content": "\n".join(text)}
            if reasoning:
                mm["reasoning_content"] = reasoning
            if calls:
                mm["tool_calls"] = calls
            msgs.append(mm)
        tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                   "parameters": t.get("input_schema", {})}}
                 for t in body.get("tools") or []]
        thinking = (body.get("thinking") or {}).get("type") == "enabled"
        oa = {"messages": msgs, "tools": tools or None, "chat_template_kwargs": {"enable_thinking": thinking}}
        prompt, thinking = chat_prompt(oa)
        max_tokens = int(body.get("max_tokens") or S.args.max_tokens)
        params = sampling_from(body)
        stops = body.get("stop_sequences") or []
        mid = "msg_" + uuid.uuid4().hex[:24]

        def to_blocks(content, reasoning, calls):
            blocks = []
            if reasoning:
                blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
            if content:
                blocks.append({"type": "text", "text": content})
            for c in calls:
                blocks.append({"type": "tool_use", "id": c["id"].replace("call_", "toolu_"), "name": c["function"]["name"],
                               "input": json.loads(c["function"]["arguments"])})
            return blocks

        stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
        if not body.get("stream"):
            content, reasoning, calls, finish, usage, _ = run(prompt, params, max_tokens, stops, thinking,
                                                              lambda k, t: None)
            return self._json(200, {"id": mid, "type": "message", "role": "assistant", "model": S.model_id,
                                    "content": to_blocks(content, reasoning, calls),
                                    "stop_reason": stop_map.get(finish, "end_turn"), "stop_sequence": None,
                                    "usage": {"input_tokens": usage["prompt_tokens"],
                                              "output_tokens": usage["completion_tokens"]}})
        self._sse_start()
        self._sse({"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant",
                                                        "model": S.model_id, "content": [], "stop_reason": None,
                                                        "usage": {"input_tokens": len(prompt), "output_tokens": 0}}},
                  "message_start")
        st = {"idx": -1, "kind": None}

        def emit(kind, txt):
            if not txt:
                return
            if st["kind"] != kind:
                if st["kind"] is not None:
                    self._sse({"type": "content_block_stop", "index": st["idx"]}, "content_block_stop")
                st["idx"] += 1
                st["kind"] = kind
                blk = {"type": "thinking", "thinking": ""} if kind == "reasoning" else {"type": "text", "text": ""}
                self._sse({"type": "content_block_start", "index": st["idx"], "content_block": blk},
                          "content_block_start")
            delta = {"type": "thinking_delta", "thinking": txt} if kind == "reasoning" else {"type": "text_delta",
                                                                                              "text": txt}
            self._sse({"type": "content_block_delta", "index": st["idx"], "delta": delta}, "content_block_delta")

        content, reasoning, calls, finish, usage, _ = run(prompt, params, max_tokens, stops, thinking, emit)
        if st["kind"] is not None:
            self._sse({"type": "content_block_stop", "index": st["idx"]}, "content_block_stop")
        for c in calls:
            st["idx"] += 1
            self._sse({"type": "content_block_start", "index": st["idx"],
                       "content_block": {"type": "tool_use", "id": c["id"].replace("call_", "toolu_"),
                                         "name": c["function"]["name"], "input": {}}}, "content_block_start")
            self._sse({"type": "content_block_delta", "index": st["idx"],
                       "delta": {"type": "input_json_delta", "partial_json": c["function"]["arguments"]}},
                      "content_block_delta")
            self._sse({"type": "content_block_stop", "index": st["idx"]}, "content_block_stop")
        self._sse({"type": "message_delta", "delta": {"stop_reason": stop_map.get(finish, "end_turn"),
                                                      "stop_sequence": None},
                   "usage": {"output_tokens": usage["completion_tokens"]}}, "message_delta")
        self._sse({"type": "message_stop"}, "message_stop")


def main():
    global S
    ap = argparse.ArgumentParser(description="bnk inference server")
    ap.add_argument("--model", required=True, help="first GGUF shard of the model")
    ap.add_argument("--mtp", default="", help="MTP draft layer GGUF (speculative decoding)")
    ap.add_argument("--draft", type=int, default=3)
    ap.add_argument("--draft-vocab", default="", help="int32 token ids the drafter may propose (faster drafts)")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--profile", default="", help="expert ranking for the initial VRAM cache")
    ap.add_argument("--counts", default="", help="file to learn expert routing counts in")
    ap.add_argument("--engine", default=str(ROOT.parent / "build" / "bnk"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("BNK_API_KEY", ""))
    ap.add_argument("--max-tokens", type=int, default=8192, help="default completion budget")
    ap.add_argument("--no-think-guard", dest="think_guard", action="store_false",
                    default=os.environ.get("BNK_THINK_GUARD", "1") != "0",
                    help="disable the thinking-loop guard (it closes reasoning that keeps repeating itself)")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--cache-dir", default="~/.cache/bnk")
    ap.add_argument("--log", default="", help="engine log file")
    ap.add_argument("--verbose", action="store_true", help="log HTTP requests")
    ap.add_argument("--log-level", choices=LEVELS, default=os.environ.get("BNK_LOG_LEVEL", "info"),
                    help="terminal output: quiet, info (requests + live status), debug (+ telemetry, engine log)")
    ap.add_argument("engine_args", nargs="*", help="extra engine arguments (after --)")
    args = ap.parse_args()
    S = State(args)
    print(f"bnk: starting the engine for {S.model_id} ...", flush=True)
    if args.log_level != "quiet":
        publish = S.engine.on_log

        def on_log(line: str):
            if not S.engine.ready.is_set():   # the engine's load progress, until it is ready
                print(f"  {line}", flush=True)
            publish(line)
        S.engine.on_log = on_log
    S.engine.start()
    info = S.engine.info
    print(f"bnk: ready - {info.get('model')} | context {info.get('n_ctx')} | MTP {'on' if info.get('mtp') else 'off'}"
          f" | thinking-loop guard {'on' if args.think_guard else 'off'}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"bnk: listening on http://{args.host}:{args.port}  (UI at /, OpenAI API at /v1)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        S.engine.stop()


if __name__ == "__main__":
    main()
