// Headline numbers derived from the live snapshot, the 1 s history and the finished requests.
import type { Live, RequestRecord, Sample } from "./telemetry"

export interface Headline {
  decode: { value: number | null; source: "live" | "last" | "none" }
  prefill: { value: number | null; source: "live" | "last" | "none" }
  accept: number | null
  tokensPerRound: number | null
  hitRate: number | null // share of expert uses served from VRAM, last minute (lifetime when idle long)
  hitRateLifetime: number | null
}

function mean(xs: (number | null | undefined)[]): number | null {
  const v = xs.filter((x): x is number => x != null && !Number.isNaN(x))
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}

export function headline(live: Live | null, history: Sample[], recent: RequestRecord[]): Headline {
  const last = recent.length ? recent[recent.length - 1] : null
  const minute = history.slice(-60)
  const r = live?.req
  let decode: Headline["decode"] = { value: null, source: "none" }
  if (live?.phase === "decode" && r && r.gen_ms > 400 && r.gen_tokens > 1) {
    decode = { value: (r.gen_tokens - 1) / (r.gen_ms / 1000), source: "live" }
  } else if (last?.tps) {
    decode = { value: last.tps, source: "last" }
  }
  let prefill: Headline["prefill"] = { value: null, source: "none" }
  if (live?.phase === "prefill" && r && r.prefill_ms > 200 && r.prefill_done > 0) {
    prefill = { value: r.prefill_done / (r.prefill_ms / 1000), source: "live" }
  } else if (last?.prefill_tps) {
    prefill = { value: last.prefill_tps, source: "last" }
  }
  const lf = live?.life
  const lifeHit = lf && lf.routed > 0 ? 1 - lf.misses / lf.routed : null
  const recentMiss = mean(minute.map((s) => s.miss_rate))
  // speculation: the current request while it runs, else the last minute, else the lifetime
  let accept: number | null = null
  let tpr: number | null = null
  if (live?.phase === "decode" && r && r.drafted > 0) {
    accept = r.accepted / r.drafted
    tpr = r.rounds > 0 ? r.gen_tokens / r.rounds : null
  } else {
    accept = mean(minute.map((s) => s.accept)) ?? (lf && lf.drafted ? lf.accepted / lf.drafted : null)
    tpr = mean(minute.map((s) => s.tokens_per_round)) ?? (lf && lf.rounds ? lf.gen_tokens / lf.rounds : null)
  }
  return {
    decode,
    prefill,
    accept,
    tokensPerRound: tpr,
    hitRate: recentMiss != null ? 1 - recentMiss : lifeHit,
    hitRateLifetime: lifeHit,
  }
}
