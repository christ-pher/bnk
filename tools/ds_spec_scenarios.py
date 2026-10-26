"""DSpark on this machine, from the sweep: round cost C(W) = c1 + m*(W-1); drafter cost D per round; per-token
acceptance a; main-model slowdown when the drafter's experts displace main experts from VRAM."""
import sys
c1, m = float(sys.argv[1]), float(sys.argv[2])          # plain round ms, marginal ms per row
cpu_share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6   # share of a plain round that is CPU expert reads
def speed(W, a, D, cache_penalty):
    exp_tok = sum(a ** k for k in range(W))
    C = (c1 + m * (W - 1)) * (1 + cpu_share * cache_penalty)
    return exp_tok / (C + D) * 1000
base = 1000 / c1
print(f"plain: {base:.1f} tok/s (round {c1:.1f} ms, +{m:.1f} ms per row)")
print("drafter placements: RAM = experts through the CPU pool (D ~12 ms, no cache loss); VRAM = experts resident (D ~3 ms,")
print("  3.9 GB of main experts evicted: production miss 20.7% -> 27.9%, i.e. CPU expert work x1.35)")
for name, D, pen in (("RAM", 12.0, 0.0), ("VRAM", 3.0, 0.35), ("free (impossible)", 0.0, 0.0)):
    print(f"\n{name}:")
    print("  a\\W " + "".join(f"{W:>9}" for W in (2, 3, 4, 5, 6)))
    for a in (0.6, 0.7, 0.8, 0.85, 0.9):
        print(f"  {a:<4}" + "".join(f"{speed(W, a, D, pen):>8.1f}x".replace("x", "") for W in (2, 3, 4, 5, 6)) + f"   (x{max(speed(W, a, D, pen) for W in (2,3,4,5,6)) / base:.2f} best)")
