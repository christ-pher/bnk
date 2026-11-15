#!/usr/bin/env python3
"""Six concurrent huge prompts through the real Engine class and a fake line-protocol engine: the stdin write
lock (fix a1d9c8d) keeps every request line whole. Without it two huge JSON lines written at the same instant
interleave in the pipe, the engine answers 'bad request' without an id (as serve.cpp does) which _read_out
drops, the request threads wait on their queues forever, and _running leaks their turn slots. The fake engine
counts every line it could not parse, and a leaked thread hangs past the join deadline - the leak's signature.
No GGUF needed. BNK_ENGINE_PY can point at another serve/engine.py to check the test catches the bug (it does:
git show a1d9c8d^:serve/engine.py > /tmp/e.py; BNK_ENGINE_PY=/tmp/e.py .venv/bin/python tests/test_engine_send_race.py
exits 1).
    .venv/bin/python tests/test_engine_send_race.py"""
import json
import os
import sys
import tempfile
import threading
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_engine_class():
    path = os.environ.get("BNK_ENGINE_PY") or str(ROOT / "serve" / "engine.py")
    spec = spec_from_file_location("bnk_engine_under_test", path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Engine


FAKE = r'''
import json, sys
bad_path = sys.argv[1]
print(json.dumps({"type": "ready", "slots": 3}), flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
    except ValueError as e:
        with open(bad_path, "a") as f:
            f.write(str(e) + "\n")
        print(json.dumps({"type": "error", "message": "bad request: " + str(e)}), flush=True)
        continue
    if req.get("cmd") == "quit":
        break
    rid = req.get("id")
    print(json.dumps({"type": "prefill", "id": rid, "ms": 1, "reused": 0}), flush=True)
    print(json.dumps({"type": "tokens", "id": rid, "ids": [1, 2, 3]}), flush=True)
    print(json.dumps({"type": "done", "id": rid, "reason": "stop", "gen_ms": 1}), flush=True)
'''

N_THREADS = 6
PROMPT = [1000] * 150_000  # ~900 KB of JSON: one write() the 64 KiB pipe cannot hold at once


def main():
    engine_cls = load_engine_class()
    fail = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fake.py").write_text(FAKE)
        bad = tmp / "bad"
        wrapper = tmp / "fake.sh"
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {tmp / "fake.py"} {bad} "$@"\n')
        wrapper.chmod(0o755)

        eng = engine_cls(str(wrapper), [])
        eng.start(timeout=30)
        done, errors = [], []

        def worker():
            try:
                for ev in eng.generate(PROMPT, {"max_tokens": 4}):
                    if ev.get("type") == "done":
                        done.append(ev)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        time.sleep(0.3)  # all six _send calls land on the pipe at one instant

        deadline = time.monotonic() + 120
        for t in threads:
            t.join(timeout=max(0.1, deadline - time.monotonic()))
        leaked = [t for t in threads if t.is_alive()]
        nbad = bad.read_text().count("\n") if bad.exists() else 0
        running = eng._running
        eng.stop()

    if nbad:
        fail.append(f"the fake engine could not parse {nbad} request line(s): the stdin writes interleaved")
    if len(done) != N_THREADS:
        fail.append(f"only {len(done)} of {N_THREADS} requests completed" + (f", {len(errors)} raised: {errors}" if errors else ""))
    if leaked:
        fail.append(f"{len(leaked)} request thread(s) never exited - the leak's signature (old bug)")
    if running != 0:
        fail.append(f"_running ended at {running}, not 0 - turn slots leaked")

    if fail:
        for f_ in fail:
            print("FAIL:", f_)
        return 1
    print(f"ok: {len(done)}/{N_THREADS} concurrent requests completed, 0 corrupt lines, _running back to 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
