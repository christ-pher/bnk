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
        # up to slots() generations at once (the engine batches them), first come first served (a request that
        # yields its turn queues behind the ones already waiting)
        self._turn = threading.Condition()
        self._line: collections.deque = collections.deque()
        self._running = 0     # requests the engine is serving now (up to its conversation slots, batched)
        self.qlock = threading.Lock()
        self.started_at = None
        self.log_tail = collections.deque(maxlen=400)
        self.env = env
        self.on_telemetry = on_telemetry   # called with every telemetry snapshot
        self.on_log = on_log               # called with every engine log line
        self.on_restart = None             # called with the exit code when the engine died and is restarted
        self.last_telemetry: dict = {}
        self.stopping = False
        self.restarts = 0

    def start(self, timeout: float = 900):
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        # each process has its own readiness: the readers of one that died cannot mark a new one ready
        ready = self.ready = threading.Event()
        proc = self.proc = subprocess.Popen([self.exe, "serve", *self.args], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=env)
        self.started_at = time.time()
        threading.Thread(target=self._read_out, args=(proc, ready), daemon=True).start()
        threading.Thread(target=self._read_err, args=(proc, ready, log), daemon=True).start()
        if not ready.wait(timeout):
            raise EngineError("the engine did not become ready")
        if proc.poll() is not None:
            raise EngineError("the engine exited during startup:\n" + "\n".join(list(self.log_tail)[-20:]))

    def _restart(self, proc):
        """The engine died while serving (e.g. it exits when the GPU context is lost): start a new one. Parked
        conversations are gone, so their next turns read their prompts again."""
        code = proc.wait()
        while not self.stopping:
            self.restarts += 1
            if self.on_restart:
                self.on_restart(code)
            try:
                self.start()
                return
            except Exception as e:   # e.g. the GPU is not usable yet: try again shortly
                self.log_tail.append(f"bnk: engine restart failed: {e}")
                time.sleep(10)

    def _read_err(self, proc, ready, log):
        for line in iter(proc.stderr.readline, b""):
            s = line.decode("utf-8", errors="replace").rstrip()
            self.log_tail.append(s)
            if "the GPU context is lost" in s:   # exiting: no new requests to it
                ready.clear()
            if self.on_log:
                self.on_log(s)
            if log is not subprocess.DEVNULL:
                log.write(line)
                log.flush()
        ready.set()

    def _read_out(self, proc, ready):
        served = False   # this process became ready
        for line in iter(proc.stdout.readline, b""):
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "ready":
                self.info = msg
                served = True
                ready.set()
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
        ready.set()
        with self.qlock:
            for q in self.queues.values():
                q.put({"type": "error", "message": "the engine process exited (restarting it)", "exited": True})
        if served and not self.stopping:   # died after startup: bring up a new one
            threading.Thread(target=self._restart, args=(proc,), daemon=True).start()

    def _send(self, obj):
        if self.proc is None or self.proc.poll() is not None:
            raise EngineError("the engine is not running")
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            self.proc.stdin.flush()
        except OSError as e:   # it is exiting (a broken pipe would read as a dropped client upstream)
            raise EngineError(f"the engine is not running ({e})") from None

    def _wait_engine(self, dead=None, timeout: float = 900) -> bool:
        """Waits for a running, ready engine (not `dead`, a process that failed): while one is restarted."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not dead and self.alive() and self.ready.is_set():
                return True
            time.sleep(0.5)
        return False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stats(self) -> dict:
        """The engine's last telemetry snapshot (it pushes one every 250 ms while working, 1 s while idle)."""
        return self.last_telemetry

    def slots(self) -> int:
        """Requests the engine serves at once (its conversation slots; decoded together in batched rounds)."""
        return max(1, int(self.info.get("slots", 1) or 1))

    def _acquire(self):
        me = object()
        with self._turn:
            self._line.append(me)
            while self._running >= self.slots() or self._line[0] is not me:
                self._turn.wait()
            self._line.popleft()
            self._running += 1

    def _release(self):
        with self._turn:
            self._running -= 1
            self._turn.notify_all()

    def waiters(self) -> int:
        """Requests waiting for the engine."""
        return len(self._line)

    def generate(self, prompt: list[int], params: dict, on_start=None):
        """Yields engine events for one request: prefill, tokens..., done (or error). `on_start(rid)` is called
        when the request gets the engine (after waiting its turn). A request the engine took but had not started
        when it died (it was exiting) goes to the engine that replaces it."""
        self._acquire()
        rid, used = None, None   # used: the process the first attempt went to
        try:
            for attempt in range(2):
                rid = uuid.uuid4().hex[:12]   # a new id per attempt: a dead process's reader posts to the ids it knew
                q = queue.Queue()
                with self.qlock:
                    self.queues[rid] = q
                if not self._wait_engine(dead=used):   # while one is restarted
                    raise EngineError("the engine is not running")
                used = self.proc
                if on_start:
                    on_start(rid)
                try:
                    self._send({"op": "generate", "id": rid, "prompt": prompt, **params})
                except EngineError:
                    if attempt:
                        raise
                    continue
                started = False
                while True:
                    msg = q.get()
                    if msg.get("exited") and not started and not attempt:
                        break   # retry
                    started = True
                    msg["_rid"] = rid
                    yield msg
                    if msg.get("type") in ("done", "error"):
                        return
                with self.qlock:
                    self.queues.pop(rid, None)
        finally:
            self._release()
            with self.qlock:
                self.queues.pop(rid, None)

    def cancel(self, rid: str):
        try:
            self._send({"op": "cancel", "id": rid})
        except EngineError:
            pass

    def stop(self):
        self.stopping = True
        if self.alive():
            try:
                self._send({"op": "quit"})
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
