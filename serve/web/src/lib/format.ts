export const fmt = {
  n(v: number | null | undefined, digits = 0) {
    if (v == null || Number.isNaN(v)) return "—"
    return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
  },
  pct(v: number | null | undefined, digits = 0) {
    if (v == null || Number.isNaN(v)) return "—"
    return `${(v * 100).toFixed(digits)}%`
  },
  compact(v: number | null | undefined) {
    if (v == null || Number.isNaN(v)) return "—"
    return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(v)
  },
  // context sizes in tokens, binary thousands as models quote them (262144 -> "256K")
  ctx(v: number | null | undefined) {
    if (v == null || Number.isNaN(v)) return "—"
    if (v < 1024) return String(v)
    const k = v / 1024
    return `${k >= 100 || Number.isInteger(k) ? Math.round(k) : k.toFixed(1)}K`
  },
  gib(mb: number | null | undefined, digits = 1) {
    if (mb == null) return "—"
    return `${(mb / 1024).toFixed(digits)} GiB`
  },
  ms(v: number | null | undefined) {
    if (v == null || Number.isNaN(v)) return "—"
    if (v >= 10000) return `${(v / 1000).toFixed(1)} s`
    if (v >= 1000) return `${(v / 1000).toFixed(2)} s`
    return `${v.toFixed(v < 10 ? 1 : 0)} ms`
  },
  duration(s: number) {
    if (!Number.isFinite(s)) return "—"
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60)
    if (d) return `${d}d ${h}h`
    if (h) return `${h}h ${m}m`
    if (m) return `${m}m ${Math.floor(s % 60)}s`
    return `${Math.floor(s)}s`
  },
  clock(t: number) {
    return new Date(t * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  },
  ago(t: number) {
    const s = Date.now() / 1000 - t
    if (s < 5) return "just now"
    if (s < 60) return `${Math.floor(s)}s ago`
    if (s < 3600) return `${Math.floor(s / 60)}m ago`
    return `${Math.floor(s / 3600)}h ago`
  },
}
