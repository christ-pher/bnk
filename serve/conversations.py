"""Which conversation each request belongs to, for the dashboard (several agents taking turns on one engine).

A request continues a conversation when its prompt shares that conversation's tokens up to at least the start of
its last turn: the same rule the engine uses to resume a parked conversation (a client may re-render the last
answer, never what came before it). Anything else starts a new conversation. Two agents with the same system
prompt stay apart, since neither shares the other's turns.
"""
from __future__ import annotations

import array
import collections
import threading
import time

MAX_CONVERSATIONS = 32
MAX_TURNS = 20
LABEL_TOKENS = 64   # tokens of a new conversation's first user message decoded for its label


def _common(a: bytes, b: bytes) -> int:
    """Length (in int32 items) of the common prefix of two packed token arrays."""
    n = min(len(a), len(b)) // 4
    if a[: n * 4] == b[: n * 4]:
        return n
    lo, hi = 0, n   # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[: mid * 4] == b[: mid * 4]:
            lo = mid
        else:
            hi = mid
    return lo


class Conversation:
    def __init__(self, cid: int, label: str):
        self.id = cid
        self.label = label
        self.created = time.time()
        self.tokens = b""            # packed int32: the last prompt plus what was generated
        self.turn_start = 0          # where the last prompt's final turn begins
        self.status = "idle"         # queued | reading | generating | idle
        self.since = self.created    # when the status last changed
        self.rid = None              # the engine request in flight (matches live.req.id)
        self.prompt_tokens = 0
        self.turns: collections.deque = collections.deque(maxlen=MAX_TURNS)
        self.n_turns = 0
        self.gen_total = 0
        self.read_ms_total = 0.0
        self.last_active = self.created

    def view(self, on_gpu, park_min: int, parking: bool) -> dict:
        n = len(self.tokens) // 4
        if self.status in ("reading", "generating"):
            where = "gpu"
        elif self.id in on_gpu:
            where = "gpu"
        elif parking and n >= park_min and self.n_turns:
            where = "ram"
        else:
            where = "none"   # the next turn reads the conversation again
        return {"id": self.id, "label": self.label, "created": self.created, "status": self.status,
                "since": self.since, "rid": self.rid, "context": n, "prompt_tokens": self.prompt_tokens,
                "where": where, "turns": list(self.turns), "n_turns": self.n_turns, "gen_total": self.gen_total,
                "read_ms_total": self.read_ms_total, "last_active": self.last_active}


class Conversations:
    def __init__(self, tok, im_start: int | None, publish, park_min: int = 2048, parking: bool = True):
        self.tok, self.im_start, self.publish = tok, im_start, publish
        self.park_min, self.parking = park_min, parking
        self.items: collections.OrderedDict[int, Conversation] = collections.OrderedDict()
        self.lock = threading.Lock()
        self.next_id = 1
        self.slots = 1
        self.on_gpu: collections.deque = collections.deque(maxlen=1)   # whose states the engine's slots hold

    def _label(self, ids: list[int]) -> str:
        """The start of the first user message (an agent's task), else of the prompt."""
        starts = [i for i, t in enumerate(ids) if t == self.im_start] if self.im_start is not None else []
        for role in ("user", "system"):
            for i in starts:
                text = self.tok.decode(ids[i + 1:i + 1 + LABEL_TOKENS])
                if text.startswith(role + "\n"):
                    s = " ".join(text[len(role) + 1:].split("<|im_end|>")[0].split())
                    if s:
                        return s[:120]
        return " ".join(self.tok.decode(ids[:LABEL_TOKENS]).split())[:120] or "(empty)"

    def begin(self, prompt_ids: list[int]) -> int:
        """Assigns a request to a conversation (creating one when none continues) and marks it queued."""
        packed = array.array("i", prompt_ids).tobytes()
        with self.lock:
            best, best_c = None, -1
            for c in self.items.values():
                if c.status != "idle" or not c.tokens:
                    continue   # a conversation has one request at a time
                n = _common(c.tokens, packed)
                if n >= c.turn_start and n > best_c:
                    best, best_c = c, n
            if best is None:
                best = Conversation(self.next_id, self._label(prompt_ids))
                self.next_id += 1
                self.items[best.id] = best
                while len(self.items) > MAX_CONVERSATIONS:
                    self.items.popitem(last=False)
            self.items.move_to_end(best.id)
            best.tokens = packed
            best.prompt_tokens = len(prompt_ids)
            last = len(prompt_ids) - 1 - prompt_ids[::-1].index(self.im_start) if self.im_start in prompt_ids else 0
            best.turn_start = max(0, last)
            self._set(best, "queued")
            return best.id

    def _set(self, c: Conversation, status: str, rid: str | None = None):
        c.status, c.since = status, time.time()
        c.rid = rid
        c.last_active = c.since
        if status in ("reading", "generating"):
            if self.on_gpu.maxlen != self.slots:
                self.on_gpu = collections.deque(self.on_gpu, maxlen=self.slots)
            if c.id in self.on_gpu:
                self.on_gpu.remove(c.id)
            self.on_gpu.append(c.id)

    def status(self, cid: int, status: str, rid: str | None = None):
        with self.lock:
            c = self.items.get(cid)
            if c:
                self._set(c, status, rid)
        self._push()

    def end(self, cid: int, generated: list[int], rec: dict):
        """A finished request: its tokens extend the conversation, and its numbers become the last turn."""
        with self.lock:
            c = self.items.get(cid)
            if c is None:
                return
            c.tokens += array.array("i", generated).tobytes()
            c.n_turns += 1
            c.gen_total += rec.get("gen_tokens") or 0
            c.read_ms_total += rec.get("prefill_ms") or 0
            c.turns.append({k: rec.get(k) for k in ("id", "time", "prompt_tokens", "reused", "prefill_ms",
                                                     "gen_tokens", "tps", "slices", "finish", "wall_s")})
            self._set(c, "idle")
        self._push()

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [c.view(set(self.on_gpu), self.park_min, self.parking) for c in reversed(self.items.values())]

    def _push(self):
        self.publish("conversations", self.snapshot())
