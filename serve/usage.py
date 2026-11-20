"""Token usage across server runs: totals per day and per model, kept in SQLite (<cache-dir>/usage.sqlite).

Every server process adds to the same file, so the totals cover every model served on this machine. A model is
its config name (configs/<name>.json: pi and qwen27b share a GGUF but count apart), else the served model id.

Counts are taken as the engine reports them, not when a request finishes: the prompt when its read starts, the
output as each batch of tokens decodes. A request the client abandons halfway still counts what it cost.
They gather in memory and go to disk every few seconds (and at exit), one transaction per flush.

    input   prompt tokens (the whole prompt, cached part included)
    cached  of those, the ones reused from the engine's cache (not read again)
    output  generated tokens
"""
from __future__ import annotations

import atexit
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    day      TEXT NOT NULL,             -- local date, YYYY-MM-DD
    model    TEXT NOT NULL,             -- config name, else the model id
    model_id TEXT NOT NULL DEFAULT '',  -- the served model id (the GGUF's name)
    requests INTEGER NOT NULL DEFAULT 0,
    input    INTEGER NOT NULL DEFAULT 0,
    cached   INTEGER NOT NULL DEFAULT 0,
    output   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model)
)"""
FIELDS = ("requests", "input", "cached", "output")


class Usage:
    FLUSH_S = 5.0

    def __init__(self, path: str, model: str, model_id: str):
        self.path, self.model, self.model_id = path, model, model_id
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.Lock()
        self.pending: dict[str, dict[str, int]] = {}   # day -> field -> count, not yet on disk
        with self._db() as db:
            db.execute(SCHEMA)
        threading.Thread(target=self._flusher, daemon=True, name="usage-flush").start()
        atexit.register(self.flush)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")   # several servers (ports) may write at once
        return db

    def add(self, **counts: int):
        day = date.today().isoformat()
        with self.lock:
            p = self.pending.setdefault(day, dict.fromkeys(FIELDS, 0))
            for k, v in counts.items():
                p[k] += int(v)

    def flush(self):
        with self.lock:
            pending, self.pending = self.pending, {}
        if not pending:
            return
        try:
            with self._db() as db:
                for day, p in pending.items():
                    db.execute("INSERT INTO usage (day, model, model_id, requests, input, cached, output) "
                               "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(day, model) DO UPDATE SET "
                               "model_id=excluded.model_id, requests=requests+excluded.requests, "
                               "input=input+excluded.input, cached=cached+excluded.cached, output=output+excluded.output",
                               (day, self.model, self.model_id, *(p[k] for k in FIELDS)))
        except sqlite3.Error as e:   # keep the counts for the next try rather than lose them
            print(f"bnk: usage: could not write {self.path}: {e}", flush=True)
            with self.lock:
                for day, p in pending.items():
                    q = self.pending.setdefault(day, dict.fromkeys(FIELDS, 0))
                    for k in FIELDS:
                        q[k] += p[k]

    def _flusher(self):
        while True:
            time.sleep(self.FLUSH_S)
            self.flush()

    def report(self, days: int = 371) -> dict:
        """Totals, per model, and per day (each model's share too) over the last `days` days; counts not yet on
        disk are included."""
        self.flush()
        with self._db() as db:
            rows = db.execute("SELECT day, model, model_id, requests, input, cached, output FROM usage").fetchall()
        since = (date.today() - timedelta(days=days - 1)).isoformat()
        total = dict.fromkeys(FIELDS, 0)
        models: dict[str, dict] = {}
        daily: dict[str, dict] = {}
        for day, model, model_id, *vals in rows:
            c = dict(zip(FIELDS, vals))
            for k in FIELDS:
                total[k] += c[k]
            m = models.setdefault(model, {"model": model, "model_id": model_id, **dict.fromkeys(FIELDS, 0),
                                          "first": day, "last": day, "days": 0})
            for k in FIELDS:
                m[k] += c[k]
            m["first"], m["last"] = min(m["first"], day), max(m["last"], day)
            m["days"] += 1
            if model_id:
                m["model_id"] = model_id
            if day >= since:
                d = daily.setdefault(day, {"day": day, **dict.fromkeys(FIELDS, 0), "models": {}})
                for k in FIELDS:
                    d[k] += c[k]
                d["models"][model] = c
        return {"total": total, "models": sorted(models.values(), key=lambda m: -(m["input"] + m["output"])),
                "daily": [daily[k] for k in sorted(daily)], "today": date.today().isoformat(),
                "current": self.model, "first": min((r[0] for r in rows), default=None),
                "generated": datetime.now().isoformat(timespec="seconds")}
