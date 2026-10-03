import { Activity, Cpu, Gauge, Layers, Zap } from "lucide-react"

import { Meter, StatCard } from "@/components/dash/stat-card"
import { Sparkline, TimeChart } from "@/components/dash/time-chart"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import { fmt } from "@/lib/format"
import { headline } from "@/lib/metrics"
import { useTelemetry, windowed, type Range } from "@/lib/telemetry"

export function OverviewPage({ range }: { range: Range }) {
  const live = useTelemetry((s) => s.live)
  const history = useTelemetry((s) => s.history)
  const recent = useTelemetry((s) => s.recent)
  const overview = useTelemetry((s) => s.overview)
  const h = headline(live, history, recent)
  const spark = history.slice(-300)
  const rows = windowed(history, range)

  return (
    <div className="flex min-h-full flex-col gap-4 md:gap-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Decode speed"
          value={fmt.n(h.decode.value, 1)}
          unit="tok/s"
          action={<SourceBadge source={h.decode.source} />}
          footer={<span>Generated tokens per second, speculation included</span>}
        >
          <Sparkline values={spark.map((s) => s.gen_tps)} color="var(--chart-1)" />
        </StatCard>
        <StatCard
          label="Prompt processing"
          value={fmt.n(h.prefill.value, 0)}
          unit="tok/s"
          action={<SourceBadge source={h.prefill.source} />}
          footer={<span>New prompt tokens per second (reused prefix excluded)</span>}
        >
          <Sparkline values={spark.map((s) => s.prefill_tps)} color="var(--chart-2)" />
        </StatCard>
        <StatCard
          label="Speculative decoding"
          value={fmt.n(h.tokensPerRound, 2)}
          unit="tok / round"
          action={live?.mtp ? <Badge variant="outline">MTP</Badge> : <Badge variant="secondary">off</Badge>}
          footer={
            <span>
              Draft acceptance <span className="font-medium text-foreground tabular">{fmt.pct(h.accept)}</span>
            </span>
          }
        >
          <Sparkline values={spark.map((s) => s.tokens_per_round)} color="var(--chart-3)" />
        </StatCard>
        <StatCard
          label="VRAM expert hit rate"
          value={fmt.pct(h.hitRate, 1)}
          footer={
            <span>
              {fmt.n(live?.experts_resident)} of {fmt.n(live?.experts_total)} experts resident · lifetime{" "}
              {fmt.pct(h.hitRateLifetime, 1)}
            </span>
          }
        >
          <Sparkline values={spark.map((s) => (s.miss_rate == null ? null : 1 - s.miss_rate))} color="var(--chart-1)" />
        </StatCard>
      </div>

      <div className="grid grid-cols-1 gap-4 md:gap-6 md:grid-cols-2 xl:grid-cols-4">
        <NowCard />
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Layers className="size-4 text-muted-foreground" />Totals</CardTitle>
            <CardDescription>Up {fmt.duration(overview?.uptime ?? 0)}</CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-x-6 gap-y-4 text-sm">
            <Total label="Requests" value={fmt.n(live?.life.requests)} />
            <Total label="Generated" value={fmt.compact(live?.life.gen_tokens)} unit="tok" />
            <Total label="Prompt" value={fmt.compact(live?.life.prompt_tokens)} unit="tok" />
            <Total label="Prefilled" value={fmt.compact(live?.life.prefill_tokens)} unit="tok" />
            <Total label="Cache swaps" value={fmt.compact(live?.life.swaps)} />
            <Total label="Context" value={`${fmt.ctx(live?.pos)} / ${fmt.ctx(live?.n_ctx)}`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Gauge className="size-4 text-muted-foreground" />GPU</CardTitle>
            <CardDescription>{overview?.gpu?.name ?? "GPU"}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <Meter label="VRAM" reading={`${fmt.gib(live?.vram_used_mb)} / ${fmt.gib(live?.vram_total_mb)}`}
              value={live ? live.vram_used_mb / live.vram_total_mb : null}
              hint={`Experts ${fmt.n(live?.expert_cache_gb, 1)} GiB · context ${fmt.n(live?.kv_gb, 2)} GiB`} />
            <Meter label="Utilization" reading={fmt.pct((live?.gpu?.util ?? 0) / 100)} value={(live?.gpu?.util ?? 0) / 100} />
            <Meter label="Power" reading={`${fmt.n(live?.gpu?.power_w)} W`}
              value={overview?.gpu?.power_limit_w && live?.gpu?.power_w != null ? live.gpu.power_w / overview.gpu.power_limit_w : null}
              hint={`${fmt.n(live?.gpu?.temp_c)} °C · SM ${fmt.n(live?.gpu?.sm_clock)} MHz`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Cpu className="size-4 text-muted-foreground" />Host</CardTitle>
            <CardDescription>CPU experts on {live?.cpu_threads ?? "—"} threads</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <Meter label="CPU" reading={fmt.pct((history.at(-1)?.cpu ?? 0) / 100)} value={(history.at(-1)?.cpu ?? 0) / 100} />
            <Meter label="RAM" reading={`${fmt.gib(live ? live.ram_total_mb - live.ram_free_mb : null)} / ${fmt.gib(live?.ram_total_mb)}`}
              value={live ? (live.ram_total_mb - live.ram_free_mb) / live.ram_total_mb : null}
              hint={`Engine resident ${fmt.gib(live?.rss_mb)} (expert store, page-locked)`} />
            <Meter label="PCIe host → GPU" reading={`${fmt.n((live?.gpu?.pcie_rx_mbs ?? 0) / 1024, 2)} GB/s`}
              value={(live?.gpu?.pcie_rx_mbs ?? 0) / 13400} hint="Measured ceiling ≈ 13.1 GB/s" />
          </CardContent>
        </Card>
      </div>

      {/* the charts take whatever height is left, so the page fills the window without scrolling */}
      <div className="grid flex-1 grid-cols-1 gap-4 md:gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Decode speed</CardTitle>
            <CardDescription>Tokens per second while generating</CardDescription>
          </CardHeader>
          <CardContent className="relative min-h-[130px] flex-1">
            <TimeChart
              rows={rows}
              series={[{ key: "gen", label: "Decode", color: "var(--chart-1)", value: (r) => r.gen_tps }]}
              unit="tok/s"
              digits={1}
              height="100%"
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Prompt processing</CardTitle>
            <CardDescription>New prompt tokens per second while reading a prompt</CardDescription>
          </CardHeader>
          <CardContent className="relative min-h-[130px] flex-1">
            <TimeChart
              rows={rows}
              series={[{ key: "pf", label: "Prefill", color: "var(--chart-2)", value: (r) => r.prefill_tps }]}
              unit="tok/s"
              height="100%"
            />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function Total({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="grid gap-0.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-lg font-semibold tabular">
        {value}
        {unit && <span className="ml-1 text-xs font-normal text-muted-foreground">{unit}</span>}
      </span>
    </div>
  )
}

function SourceBadge({ source }: { source: "live" | "last" | "none" }) {
  if (source === "live")
    return (
      <Badge variant="outline" className="gap-1.5">
        <span className="size-1.5 animate-pulse rounded-full bg-good" />
        live
      </Badge>
    )
  if (source === "last") return <Badge variant="secondary">last request</Badge>
  return null
}

// What the engine is doing right now.
export function NowCard() {
  const live = useTelemetry((s) => s.live)
  const recent = useTelemetry((s) => s.recent)
  const waiting = useTelemetry((s) => s.overview?.waiting ?? 0)
  const r = live?.req
  const last = recent.at(-1)
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Activity className="size-4 text-muted-foreground" />Now</CardTitle>
        <CardDescription>{live ? `Context ${fmt.ctx(live.pos)} / ${fmt.ctx(live.n_ctx)}` : "Connecting…"}</CardDescription>
        <CardAction>
          {live?.phase === "prefill" && <Badge className="gap-1.5"><Spinner className="size-3" />Reading prompt</Badge>}
          {live?.phase === "decode" && <Badge className="gap-1.5"><Zap className="size-3" />Generating</Badge>}
          {live?.phase === "idle" && <Badge variant="secondary">Idle</Badge>}
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-4 text-sm">
        {live?.phase === "prefill" && r && (
          <>
            <div className="grid gap-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Prompt</span>
                <span className="tabular">{fmt.n(r.prefill_done)} / {fmt.n(r.prefill_total)} tokens</span>
              </div>
              <Progress value={r.prefill_total ? (100 * r.prefill_done) / r.prefill_total : 0} />
            </div>
            <Row label="Elapsed" value={fmt.ms(r.prefill_ms)} />
            <Row label="Rate" value={r.prefill_ms > 200 ? `${fmt.n(r.prefill_done / (r.prefill_ms / 1000))} tok/s` : "—"} />
          </>
        )}
        {live?.phase === "decode" && r && (
          <>
            <Row label="Generated" value={`${fmt.n(r.gen_tokens)} tokens`} />
            <Row label="Speed" value={r.gen_ms > 400 ? `${fmt.n((r.gen_tokens - 1) / (r.gen_ms / 1000), 1)} tok/s` : "—"} />
            <Row label="Tokens per round" value={r.rounds ? fmt.n(r.gen_tokens / r.rounds, 2) : "—"} />
            <Row label="Draft acceptance" value={r.drafted ? fmt.pct(r.accepted / r.drafted) : "—"} />
            <Row label="Expert misses (CPU)" value={r.routed ? fmt.pct(r.misses / r.routed, 1) : "—"} />
            <Row label="Prompt" value={`${fmt.n(r.prompt_tokens)} tokens${r.reused ? ` · ${fmt.n(r.reused)} reused` : ""}`} />
          </>
        )}
        {live?.phase === "idle" && (
          <>
            <p className="text-muted-foreground">Waiting for requests{waiting ? ` · ${waiting} queued` : ""}.</p>
            {last && (
              <>
                <Separator />
                <div className="grid gap-2">
                  <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Last request · {fmt.ago(last.time)}</span>
                  <Row label="Decode" value={`${fmt.n(last.tps, 1)} tok/s · ${fmt.n(last.gen_tokens)} tokens`} />
                  <Row label="Prompt" value={`${fmt.n(last.prefill_tps)} tok/s · ${fmt.n(last.prompt_tokens)} tokens`} />
                  <Row label="Tokens per round" value={fmt.n(last.tokens_per_round, 2)} />
                  <Row label="Expert misses" value={fmt.pct(last.expert_miss_rate, 1)} />
                </div>
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium tabular">{value}</span>
    </div>
  )
}
