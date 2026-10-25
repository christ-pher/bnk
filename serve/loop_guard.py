"""Loop detection: notices when a model's output keeps repeating itself.

Some models (Orca in particular) can fall into a reasoning loop, regenerating the same passage until the output
budget runs out. One guard watches the thinking tokens. Once at least `min_tokens` have been generated, every
`every` tokens it looks at the last `window` tokens and counts how many of their `span`-token stretches already
appeared earlier in that window. When the share reaches `threshold`, the reasoning is looping, and the server
closes it (WRAP_UP) so the model answers with what it has.

A second instance (content_guard) watches the answer: DeepSeek-V4.1 was seen repeating a tool call verbatim for
27,000 tokens until max_tokens (2026-10-25). Answers are legitimately more repetitive than reasoning (code, tables),
so its bar is higher and it starts later; when it fires the server ends the request as `length` and drops the
tool call that was being repeated.

Exact-token repetition only: a loop that paraphrases itself is not caught, and legitimately repetitive reasoning
(long tables, enumerations) can trip it, which is why the bar is high and the window long. The cost of a check is
bounded by the window, whatever the length of the reasoning.
"""
from __future__ import annotations

from array import array
from collections import deque


class ThinkingLoopGuard:
    # 12-token spans catch loops whose lines vary slightly (a counter, a number); the 4K window catches cycles of up
    # to ~2K tokens. On real reasoning traces the repeated share stays under 5%.
    def __init__(self, window: int = 4096, min_tokens: int = 1536, every: int = 64, span: int = 12,
                 threshold: float = 0.55):
        self.window, self.min_tokens, self.every, self.span, self.threshold = window, min_tokens, every, span, threshold
        self.tokens: deque[int] = deque(maxlen=window)
        self.count = 0
        self.last_ratio = 0.0

    def push(self, ids) -> bool:
        """Feed newly generated thinking tokens; True when the reasoning is looping."""
        hit = False
        for t in ids:
            self.tokens.append(t)
            self.count += 1
            if self.count >= self.min_tokens and self.count % self.every == 0 and self.check():
                hit = True
        return hit

    def check(self) -> bool:
        raw = array("i", self.tokens).tobytes()
        w = 4 * self.span
        n = len(self.tokens) - self.span + 1
        if n <= 0:
            return False
        seen: set[bytes] = set()
        repeated = 0
        for i in range(0, 4 * n, 4):
            key = raw[i:i + w]
            if key in seen:
                repeated += 1
            else:
                seen.add(key)
        self.last_ratio = repeated / n
        return self.last_ratio >= self.threshold


def content_guard() -> ThinkingLoopGuard:
    """The answer's guard: later and stricter than the thinking one (see the module docstring)."""
    return ThinkingLoopGuard(min_tokens=2048, threshold=0.65)


# What the model reads when the guard steps in: it is told to stop and answer, and the reasoning is closed.
WRAP_UP = ("\n\nI notice I am going around in circles and repeating the same reasoning. I already have what I need, "
           "so I will stop thinking here and give my answer.\n</think>\n\n")


def at_clean_boundary(text: str) -> bool:
    """A good place to cut in: the end of a line or of a sentence."""
    t = text.rstrip(" ")
    return t.endswith(("\n", ".", "!", "?", ":", ";"))
