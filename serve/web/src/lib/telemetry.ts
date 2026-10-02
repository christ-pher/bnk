// The live telemetry store: one EventSource on /api/stream feeding every view.
//
// The server sends `init` (everything it has), then `live` (~4/s, the engine's latest snapshot), `sample`
// (1/s, rates computed server-side from cumulative counters), `request` (a finished request) and `log`.
import { useSyncExternalStore } from "react"

export type Phase = "idle" | "prefill" | "decode"

export interface Live {
  t: number
  model: string
  phase: Phase
  n_ctx: number
  pos: number
  n_layer: number
  n_expert: number
  n_expert_used: number
  vram_total_mb: number
  vram_used_mb: number
  ram_total_mb: number
  ram_free_mb: number
  rss_mb: number
  experts_resident: number
  experts_total: number
  expert_cache_gb: number
  cpu_threads: number
  mtp: boolean
  req: {
    id: string
    prompt_tokens: number
    reused: number
    prefill_done: number
    prefill_total: number
    prefill_ms: number
    gen_tokens: number
    gen_ms: number
    elapsed_ms: number
    rounds: number
    drafted: number
    accepted: number
    routed: number
    misses: number
  }
  life: {
    requests: number
    prompt_tokens: number
    prefill_tokens: number
    prefill_ms: number
    gen_tokens: number
    gen_ms: number
    rounds: number
    drafted: number
    accepted: number
    verify_ms: number
    draft_ms: number
    cpu_expert_ms: number
    routed: number
    misses: number
    swaps: number
  }
  layer_slots: number[]
  layer_routed: number[]
  layer_misses: number[]
  gpu?: Gpu
  cores?: number[]
}

export interface Gpu {
  t?: number
  util?: number | null
  mem_util?: number | null
  power_w?: number | null
  temp_c?: number | null
  mem_temp_c?: number | null
  sm_clock?: number | null
  mem_clock?: number | null
  pcie_rx_mbs?: number | null
  pcie_tx_mbs?: number | null
}

export interface Sample {
  t: number
  phase: Phase
  gen_tps: number | null // tokens per second of decoding time (null: no decoding in this second)
  prefill_tps: number | null
  gen_tokens: number // tokens produced in this second
  prefill_tokens: number
  accept: number | null
  tokens_per_round: number | null
  miss_rate: number | null
  swaps: number
  busy: number
  vram_used_mb: number
  ram_used_mb: number
  cpu: number
  gpu_util: number | null
  gpu_power_w: number | null
  gpu_temp_c: number | null
  pcie_rx_mbs: number | null
  pcie_tx_mbs: number | null
  pos: number
}

export interface RequestRecord {
  id: string
  time: number
  api: string
  prompt_tokens: number
  gen_tokens: number
  finish: string
  temperature?: number
  max_tokens?: number
  thinking?: boolean
  prefill_ms?: number
  prefill_tps?: number
  gen_ms?: number
  tps?: number
  reused?: number
  rounds?: number
  tokens_per_round?: number
  accepted?: number
  drafted?: number
  expert_miss_rate?: number
  verify_ms?: number
  draft_ms?: number
  cpu_expert_ms?: number
}

export interface Overview {
  model: string
  uptime: number
  started: number
  alive: boolean
  waiting: number
  defaults: { temperature: number; top_k: number; top_p: number }
  engine: { n_ctx: number; n_vocab: number; mtp: boolean; name: string; args: string[] }
  gpu: { name?: string; power_limit_w?: number; sm_clock_max?: number; pcie_gen?: number; pcie_width?: number; driver?: string }
  live: Live
  history: Sample[]
  recent: RequestRecord[]
  log: string[]
}

export type Connection = "connecting" | "open" | "down"

export interface State {
  conn: Connection
  overview: Overview | null
  live: Live | null
  history: Sample[]
  recent: RequestRecord[]
  log: { t: number; line: string }[]
  lastEvent: number
}

const MAX_HISTORY = 3600
const MAX_LOG = 1000

let state: State = { conn: "connecting", overview: null, live: null, history: [], recent: [], log: [], lastEvent: 0 }
const listeners = new Set<() => void>()
let es: EventSource | null = null
let retry: number | undefined

function set(patch: Partial<State>) {
  state = { ...state, ...patch, lastEvent: Date.now() }
  listeners.forEach((l) => l())
}

function connect() {
  es?.close()
  es = new EventSource("/api/stream")
  es.addEventListener("init", (e) => {
    const o = JSON.parse((e as MessageEvent).data) as Overview
    set({
      conn: "open",
      overview: o,
      live: o.live && "phase" in o.live ? o.live : state.live,
      history: o.history.slice(-MAX_HISTORY),
      recent: o.recent,
      log: o.log.map((line) => ({ t: 0, line })),
    })
  })
  es.addEventListener("live", (e) => {
    const l = JSON.parse((e as MessageEvent).data) as Live
    const lastSnap = layerSnaps[layerSnaps.length - 1]
    if (l.layer_routed && (!lastSnap || l.t - lastSnap.t >= 1)) {
      layerSnaps.push({ t: l.t, routed: l.layer_routed, misses: l.layer_misses })
      if (layerSnaps.length > 3700) layerSnaps.shift()
    }
    set({ conn: "open", live: l })
  })
  es.addEventListener("sample", (e) => {
    const s = JSON.parse((e as MessageEvent).data) as Sample
    const h = state.history.length >= MAX_HISTORY ? state.history.slice(1) : state.history.slice()
    h.push(s)
    set({ history: h })
  })
  es.addEventListener("request", (e) => {
    const r = JSON.parse((e as MessageEvent).data) as RequestRecord
    set({ recent: [...state.recent.slice(-199), r] })
  })
  es.addEventListener("log", (e) => {
    const l = JSON.parse((e as MessageEvent).data) as { t: number; line: string }
    set({ log: [...state.log.slice(-(MAX_LOG - 1)), l] })
  })
  es.onerror = () => {
    // EventSource retries by itself; after a hard failure (server restarting) reconnect with a fresh `init`
    set({ conn: "down" })
    if (es?.readyState === EventSource.CLOSED) {
      clearTimeout(retry)
      retry = window.setTimeout(connect, 2000)
    }
  }
}

function subscribe(l: () => void) {
  listeners.add(l)
  if (!es) connect()
  return () => listeners.delete(l)
}

export function useTelemetry<T>(select: (s: State) => T): T {
  return useSyncExternalStore(subscribe, () => select(state))
}

// Per-layer routing counters, one snapshot a second, for windowed per-layer miss rates.
const layerSnaps: { t: number; routed: number[]; misses: number[] }[] = []

// Per-layer (uses, CPU misses) over the last `seconds` (or as far back as this page has seen); null when no
// expert was used in that window.
export function layerWindow(live: Live, seconds: number | null): { routed: number[]; misses: number[] } {
  if (seconds == null || !layerSnaps.length) return { routed: live.layer_routed, misses: live.layer_misses }
  const t0 = live.t - seconds
  let base = layerSnaps[0]
  for (const s of layerSnaps) {
    if (s.t <= t0) base = s
    else break
  }
  return {
    routed: live.layer_routed.map((v, i) => v - (base.routed[i] ?? 0)),
    misses: live.layer_misses.map((v, i) => v - (base.misses[i] ?? 0)),
  }
}

// Seconds a time window covers, for the range selector shared by every chart.
export const RANGES = { "1m": 60, "5m": 300, "15m": 900, "1h": 3600 } as const
export type Range = keyof typeof RANGES

export function windowed(history: Sample[], range: Range): Sample[] {
  if (!history.length) return history
  const t1 = history[history.length - 1].t
  const t0 = t1 - RANGES[range]
  let i = history.length - 1
  while (i > 0 && history[i - 1].t >= t0) i--
  return history.slice(i)
}

// Exponential smoothing for the headline numbers (charts keep the raw 1 s samples).
export function smooth(values: (number | null | undefined)[], alpha = 0.35): number | null {
  let v: number | null = null
  for (const x of values) {
    if (x == null || Number.isNaN(x)) continue
    v = v == null ? x : alpha * x + (1 - alpha) * v
  }
  return v
}
