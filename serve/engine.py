"""The bnk engine as a child process speaking the line protocol (src/server/serve.cpp)."""
from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import threading
import time
import uuid


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, exe: str, args: list[str], log_path: str | None = None, env: dict | None = None,
                 on_telemetry=None, on_log=None):
        self.exe, self.args, self.log_path = exe, args, log_path
        self.proc = None
        self.ready = threading.Event()
        self.info = {}
        self.queues: dict[str, queue.Queue] = {}
        self.lock = threading.Lock()          # one generation at a time
        self.qlock = threading.Lock()
        self.started_at = None
        self.log_tail = collections.deque(maxlen=400)
        self.env = env
        self.on_telemetry = on_telemetry   # called with every telemetry snapshot
        self.on_log = on_log               # called with every engine log line
        self.last_telemetry: dict = {}

    def start(self, timeout: float = 900):
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        self.proc = subprocess.Popen([self.exe, "serve", *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0, env=env)
        self.started_at = time.time()
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, args=(log,), daemon=True).start()
        if not self.ready.wait(timeout):
            raise EngineError("the engine did not become ready")
        if self.proc.poll() is not None:
            raise EngineError("the engine exited during startup:\n" + "\n".join(list(self.log_tail)[-20:]))

    def _read_err(self, log):
        for line in iter(self.proc.stderr.readline, b""):
            s = line.decode("utf-8", errors="replace").rstrip()
            self.log_tail.append(s)
            if self.on_log:
                self.on_log(s)
            if log is not subprocess.DEVNULL:
                log.write(line)
                log.flush()
        self.ready.set()

    def _read_out(self):
        for line in iter(self.proc.stdout.readline, b""):
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "ready":
                self.info = msg
                self.ready.set()
                continue
            if t == "telemetry":
                self.last_telemetry = msg
                if self.on_telemetry:
                    self.on_telemetry(msg)
                continue
            key = msg.get("id") or "_misc"
            with self.qlock:
                q = self.queues.get(key)
            if q is not None:
                q.put(msg)
        self.ready.set()
        with self.qlock:
            for q in self.queues.values():
                q.put({"type": "error", "message": "the engine process exited"})

    def _send(self, obj):
        if self.proc is None or self.proc.poll() is not None:
            raise EngineError("the engine is not running")
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stats(self) -> dict:
        """The engine's last telemetry snapshot (it pushes one every 250 ms while working, 1 s while idle)."""
        return self.last_telemetry

    def generate(self, prompt: list[int], params: dict):
        """Yields engine events for one request: prefill, tokens..., done (or error)."""
        rid = uuid.uuid4().hex[:12]
        q = queue.Queue()
        with self.qlock:
            self.queues[rid] = q
        try:
            with self.lock:
                self._send({"op": "generate", "id": rid, "prompt": prompt, **params})
                while True:
                    msg = q.get()
                    msg["_rid"] = rid
                    yield msg
                    if msg.get("type") in ("done", "error"):
                        break
        finally:
            with self.qlock:
                self.queues.pop(rid, None)

    def cancel(self, rid: str):
        try:
            self._send({"op": "cancel", "id": rid})
        except EngineError:
            pass

    def stop(self):
        if self.alive():
            try:
                self._send({"op": "quit"})
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
