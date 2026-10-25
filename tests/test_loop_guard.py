#!/usr/bin/env python3
"""The loop guards: the thinking guard and the answer's guard (later and stricter) fire on verbatim repetition and
stay quiet on varied text.
    .venv/bin/python tests/test_loop_guard.py"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.loop_guard import ThinkingLoopGuard, content_guard  # noqa: E402

rng = random.Random(7)
fails = 0


def feed(guard, ids, chunk=7):
    """Streams ids in draft-sized chunks; -> the count at which the guard fired, or None."""
    for i in range(0, len(ids), chunk):
        if guard.push(ids[i:i + chunk]):
            return guard.count
    return None


# 1. a verbatim loop (the 2026-10-25 case: a tool call repeated for 27,000 tokens): both guards fire
cycle = [rng.randrange(1000, 100000) for _ in range(300)]
loop = cycle * 40
t = feed(ThinkingLoopGuard(), loop)
c = feed(content_guard(), loop)
ok = t is not None and t <= 2048 and c is not None and 2048 <= c <= 3072
fails += not ok
print(f"{'loop':12s} {'OK' if ok else 'FAIL'}: thinking fired at {t}, answer at {c}")

# 2. varied text (random tokens): neither fires over 20K tokens
varied = [rng.randrange(1000, 100000) for _ in range(20000)]
t, c = feed(ThinkingLoopGuard(), varied), feed(content_guard(), varied)
ok = t is None and c is None
fails += not ok
print(f"{'varied':12s} {'OK' if ok else 'FAIL'}: thinking {t}, answer {c}")

# 3. repetitive but legitimate code: 400 near-identical lines that differ by a counter token every 12 tokens;
#    the answer's guard stays quiet (the thinking guard's lower bar may not: it is not the one watching answers)
line = [rng.randrange(1000, 100000) for _ in range(11)]
code = []
for n in range(400):
    code += line[:5] + [n] + line[5:]
c = feed(content_guard(), code)
ok = c is None
fails += not ok
print(f"{'enum lines':12s} {'OK' if ok else 'FAIL'}: answer {c}")

# 4. a half-repetitive answer (every other 300-token block is new): the answer's guard stays quiet, 65% is its bar
mixed = []
for n in range(30):
    mixed += cycle if n % 2 else [rng.randrange(1000, 100000) for _ in range(300)]
c = feed(content_guard(), mixed)
ok = c is None
fails += not ok
print(f"{'half':12s} {'OK' if ok else 'FAIL'}: answer {c}")

print("FAILURES" if fails else "all passed")
sys.exit(1 if fails else 0)
