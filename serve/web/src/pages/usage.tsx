// Token usage across server runs and models: totals, a table per model, and a year of days as a heat map.
// The numbers come from /api/usage (serve/usage.py, kept in <cache-dir>/usage.sqlite), refetched whenever a
// request finishes and every 15 s while the page is open (a long generation counts as it decodes).
import { useEffect, useMemo, useState } from "react"

import { StatCard } from "@/components/dash/stat-card"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { fmt } from "@/lib/format"
import { useTelemetry } from "@/lib/telemetry"

interface Counts {
  requests: number
  input: number
  cached: number
  output: number
}
interface ModelRow extends Counts {
  model: string
  model_id: string
  first: string
  last: string
  days: number
}
interface Day extends Counts {
  day: string
  models: Record<string, Counts>
}
interface Report {
  total: Counts
  models: ModelRow[]
  daily: Day[]
  today: string
  current: string
  first: string | null
}

type Metric = "total" | "input" | "output"
const METRICS: Record<Metric, { label: string; of: (c: Counts) => number }> = {
  total: { label: "Total", of: (c) => c.input + c.output },
  input: { label: "Input", of: (c) => c.input },
  output: { label: "Output", of: (c) => c.output },
}

function useUsage() {
  const [report, setReport] = useState<Report | null>(null)
  const [error, setError] = useState<string | null>(null)
  const finished = useTelemetry((s) => s.recent.length)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 15000)
    return () => clearInterval(t)
  }, [])
  useEffect(() => {
    let live = true
    fetch("/api/usage")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((r: Report) => live && (setReport(r), setError(null)))
      .catch((e: Error) => live && setError(e.message))
    return () => {
      live = false
    }
  }, [finished, tick])
  return { report, error }
}

export function UsagePage() {
  const { report, error } = useUsage()
  const [metric, setMetric] = useState<Metric>("total")
  const [model, setModel] = useState<string>("all")

  if (!report)
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>{error ? "Usage unavailable" : "Loading usage…"}</EmptyTitle>
          {error && <EmptyDescription>/api/usage: {error}</EmptyDescription>}
        </EmptyHeader>
      </Empty>
    )

  const t = report.total
  const today = report.daily.find((d) => d.day === report.today)
  const last30 = report.daily.filter((d) => d.day >= isoDaysAgo(report.today, 29))
  const sum30 = last30.reduce((a, d) => a + d.input + d.output, 0)
  const grand = t.input + t.output

  return (
    <div className="flex min-h-full flex-col gap-4 md:gap-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Total tokens"
          value={fmt.compact(grand)}
          footer={
            <>
              <span>{fmt.n(grand)} tokens over {fmt.n(t.requests)} requests</span>
              {report.first && <span>Since {fmtDay(report.first)}</span>}
            </>
          }
        />
        <StatCard
          label="Input tokens"
          value={fmt.compact(t.input)}
          footer={
            <>
              <span>{fmt.n(t.input)} prompt tokens</span>
              <span>{fmt.pct(t.input ? t.cached / t.input : null)} reused from the cache</span>
            </>
          }
        />
        <StatCard
          label="Output tokens"
          value={fmt.compact(t.output)}
          footer={
            <>
              <span>{fmt.n(t.output)} generated tokens</span>
              <span>{t.requests ? `${fmt.n(t.output / t.requests)} per request` : "—"}</span>
            </>
          }
        />
        <StatCard
          label="Today"
          value={fmt.compact(today ? today.input + today.output : 0)}
          footer={
            <>
              <span>
                {fmt.n(today?.input ?? 0)} in · {fmt.n(today?.output ?? 0)} out · {fmt.n(today?.requests ?? 0)} requests
              </span>
              <span>Last 30 days {fmt.compact(sum30)}</span>
            </>
          }
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Daily tokens</CardTitle>
          <CardDescription>The last year, one square per day (local time of the server)</CardDescription>
          <CardAction className="flex flex-wrap items-center gap-2">
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger size="sm" className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All models</SelectItem>
                {report.models.map((m) => (
                  <SelectItem key={m.model} value={m.model}>{m.model}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <ToggleGroup type="single" variant="outline" size="sm" value={metric} onValueChange={(v) => v && setMetric(v as Metric)}>
              {(Object.keys(METRICS) as Metric[]).map((k) => (
                <ToggleGroupItem key={k} value={k} className="px-2.5">{METRICS[k].label}</ToggleGroupItem>
              ))}
            </ToggleGroup>
          </CardAction>
        </CardHeader>
        <CardContent>
          <Heatmap report={report} metric={metric} model={model} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>By model</CardTitle>
          <CardDescription>Each config counts on its own (two configs of one GGUF are two rows)</CardDescription>
        </CardHeader>
        <CardContent>
          {report.models.length === 0 ? (
            <Empty>
              <EmptyHeader>
                <EmptyTitle>No usage yet</EmptyTitle>
                <EmptyDescription>Counts start with the first request after this update.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <Table className="tabular">
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Requests</TableHead>
                  <TableHead className="text-right">Input</TableHead>
                  <TableHead className="text-right">Reused</TableHead>
                  <TableHead className="text-right">Output</TableHead>
                  <TableHead className="text-right">Total</TableHead>
                  <TableHead className="w-40">Share</TableHead>
                  <TableHead>Days</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {report.models.map((m) => {
                  const tot = m.input + m.output
                  return (
                    <TableRow key={m.model}>
                      <TableCell>
                        <div className="flex items-center gap-2 font-medium">
                          {m.model}
                          {m.model === report.current && <Badge variant="secondary">serving</Badge>}
                        </div>
                        {m.model_id && m.model_id !== m.model && <div className="text-xs text-muted-foreground">{m.model_id}</div>}
                      </TableCell>
                      <TableCell className="text-right">{fmt.n(m.requests)}</TableCell>
                      <TableCell className="text-right">{fmt.n(m.input)}</TableCell>
                      <TableCell className="text-right text-muted-foreground">{fmt.pct(m.input ? m.cached / m.input : null)}</TableCell>
                      <TableCell className="text-right">{fmt.n(m.output)}</TableCell>
                      <TableCell className="text-right font-medium">{fmt.n(tot)}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                            <div className="h-full rounded-full bg-chart-1" style={{ width: `${grand ? (tot / grand) * 100 : 0}%` }} />
                          </div>
                          <span className="w-10 text-right text-xs text-muted-foreground">{fmt.pct(grand ? tot / grand : null)}</span>
                        </div>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {m.days} · {fmtDay(m.first)} – {fmtDay(m.last)}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ---- the heat map: 53 week columns (Sunday on top), five shades by quartile of the active days
const LEVELS = ["bg-muted", "bg-chart-1/25", "bg-chart-1/50", "bg-chart-1/75", "bg-chart-1"]
const WEEKDAYS = ["", "Mon", "", "Wed", "", "Fri", ""]

function Heatmap({ report, metric, model }: { report: Report; metric: Metric; model: string }) {
  const of = METRICS[metric].of
  const { weeks, months, cuts, yearTotal, active } = useMemo(() => {
    const byDay = new Map(report.daily.map((d) => [d.day, d]))
    const value = (d?: Day) => {
      if (!d) return 0
      if (model === "all") return of(d)
      const c = d.models[model]
      return c ? of(c) : 0
    }
    const end = parseDay(report.today)
    const start = new Date(end)
    start.setDate(start.getDate() - 52 * 7 - end.getDay()) // a Sunday, 53 columns back
    const weeks: Cell[][] = []
    const months: { col: number; label: string }[] = []
    for (let cur = new Date(start), col = 0; cur <= end || cur.getDay() !== 0; col++) {
      const week = []
      for (let i = 0; i < 7; i++) {
        const iso = toIso(cur)
        const d = byDay.get(iso)
        week.push({ day: iso, date: new Date(cur), v: value(d), d, future: cur > end })
        if (cur.getDate() === 1) {
          const label = cur.toLocaleDateString(undefined, { month: "short" })
          if (!months.length || col - months[months.length - 1].col >= 3) months.push({ col, label })
        }
        cur.setDate(cur.getDate() + 1)
      }
      weeks.push(week)
    }
    const vals = weeks.flat().filter((c) => !c.future && c.v > 0).map((c) => c.v).sort((a, b) => a - b)
    const q = (p: number) => vals[Math.min(vals.length - 1, Math.floor(p * vals.length))] ?? 0
    return { weeks, months, cuts: [q(0.25), q(0.5), q(0.75)], yearTotal: vals.reduce((a, b) => a + b, 0), active: vals.length }
  }, [report, of, model])

  const level = (v: number) => (v <= 0 ? 0 : v <= cuts[0] ? 1 : v <= cuts[1] ? 2 : v <= cuts[2] ? 3 : 4)

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto pb-1">
        <div className="inline-grid gap-[3px]" style={{ gridTemplateColumns: `auto repeat(${weeks.length}, 11px)` }}>
          {/* month labels */}
          <div />
          {weeks.map((_, col) => {
            const m = months.find((x) => x.col === col)
            return (
              <div key={col} className="relative h-4 text-[10px] text-muted-foreground">
                {m && <span className="absolute left-0 whitespace-nowrap">{m.label}</span>}
              </div>
            )
          })}
          {Array.from({ length: 7 }, (_, row) => (
            <Row key={row} row={row} weeks={weeks} level={level} metric={metric} model={model} today={report.today} />
          ))}
        </div>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          {fmt.n(yearTotal)} {METRICS[metric].label.toLowerCase()} tokens in the last year · {active} active days
        </span>
        <span className="flex items-center gap-1">
          Less
          {LEVELS.map((c) => (
            <span key={c} className={`size-[11px] rounded-[2px] ${c}`} />
          ))}
          More
        </span>
      </div>
    </div>
  )
}

type Cell = { day: string; date: Date; v: number; d?: Day; future: boolean }

function Row({ row, weeks, level, metric, model, today }: {
  row: number
  weeks: Cell[][]
  level: (v: number) => number
  metric: Metric
  model: string
  today: string
}) {
  return (
    <>
      <div className="pr-1 text-right text-[10px] leading-[11px] text-muted-foreground">{WEEKDAYS[row]}</div>
      {weeks.map((w) => {
        const c = w[row]
        if (c.future) return <div key={c.day} className="size-[11px]" />
        return (
          <Tooltip key={c.day}>
            <TooltipTrigger asChild>
              <div className={`size-[11px] rounded-[2px] ${LEVELS[level(c.v)]} ${c.day === today ? "ring-1 ring-foreground/40" : ""}`} />
            </TooltipTrigger>
            <TooltipContent>
              <DayTip c={c} metric={metric} model={model} />
            </TooltipContent>
          </Tooltip>
        )
      })}
    </>
  )
}

function DayTip({ c, metric, model }: { c: { date: Date; v: number; d?: Day }; metric: Metric; model: string }) {
  const title = c.date.toLocaleDateString(undefined, { weekday: "short", year: "numeric", month: "short", day: "numeric" })
  if (!c.d || c.v === 0)
    return (
      <div className="tabular">
        <div className="font-medium">{title}</div>
        <div>No tokens{model !== "all" ? ` on ${model}` : ""}</div>
      </div>
    )
  const shown = model === "all" ? c.d : c.d.models[model]
  const per = Object.entries(c.d.models).sort((a, b) => METRICS[metric].of(b[1]) - METRICS[metric].of(a[1]))
  return (
    <div className="grid gap-1 tabular">
      <div className="font-medium">{title}</div>
      <div>
        {fmt.n(c.v)} {METRICS[metric].label.toLowerCase()} tokens
        {shown && <> · {fmt.n(shown.input)} in · {fmt.n(shown.output)} out · {fmt.n(shown.requests)} req</>}
      </div>
      {model === "all" && per.length > 1 && (
        <div className="grid gap-0.5 opacity-80">
          {per.map(([name, m]) => (
            <div key={name} className="flex justify-between gap-4">
              <span>{name}</span>
              <span>{fmt.n(METRICS[metric].of(m))}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function parseDay(s: string) {
  const [y, m, d] = s.split("-").map(Number)
  return new Date(y, m - 1, d)
}
function toIso(d: Date) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`
}
function isoDaysAgo(today: string, n: number) {
  const d = parseDay(today)
  d.setDate(d.getDate() - n)
  return toIso(d)
}
function fmtDay(s: string) {
  return parseDay(s).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
}
