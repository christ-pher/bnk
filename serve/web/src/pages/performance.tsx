import { Bar, BarChart, CartesianGrid, Scatter, ScatterChart, XAxis, YAxis, ZAxis } from "recharts"

import { TimeChart } from "@/components/dash/time-chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { fmt } from "@/lib/format"
import { useTelemetry, windowed, type Range } from "@/lib/telemetry"

export function PerformancePage({ range }: { range: Range }) {
  const history = useTelemetry((s) => s.history)
  const recent = useTelemetry((s) => s.recent)
  const life = useTelemetry((s) => s.live?.life)
  const rows = windowed(history, range)
  const reqs = recent.slice(-40).map((r, i) => ({ ...r, n: i + 1 }))
  const perRound = (v?: number) => (life && life.rounds ? (v ?? 0) / life.rounds : null)

  return (
    <div className="grid grid-cols-1 gap-4 md:gap-6 xl:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Decode speed</CardTitle>
          <CardDescription>Tokens per second while generating</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} series={[{ key: "gen", label: "Decode", color: "var(--chart-1)", value: (r) => r.gen_tps }]} unit="tok/s" digits={1} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Prompt processing</CardTitle>
          <CardDescription>New prompt tokens per second while reading a prompt</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} series={[{ key: "pf", label: "Prefill", color: "var(--chart-2)", value: (r) => r.prefill_tps }]} unit="tok/s" />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Tokens per verify round</CardTitle>
          <CardDescription>1 + accepted drafts; each round is one pass of the full model</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} kind="line" series={[{ key: "tpr", label: "Tokens / round", color: "var(--chart-3)", value: (r) => r.tokens_per_round, missing: "connect" }]} digits={2} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Draft acceptance</CardTitle>
          <CardDescription>Share of MTP drafts the full model confirmed</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} kind="line" series={[{ key: "acc", label: "Accepted", color: "var(--chart-3)", value: (r) => (r.accept == null ? null : r.accept * 100), missing: "connect" }]} unit="%" domain={[0, 100]} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Decode speed by request</CardTitle>
          <CardDescription>The last {reqs.length} requests, oldest first</CardDescription>
        </CardHeader>
        <CardContent>
          {reqs.length ? (
            <ChartContainer config={{ tps: { label: "Decode", color: "var(--chart-1)" } }} className="aspect-auto h-[220px] w-full">
              <BarChart data={reqs} margin={{ top: 8, right: 8 }} barCategoryGap={2}>
                <CartesianGrid vertical={false} strokeOpacity={0.5} />
                <XAxis dataKey="n" tickLine={false} axisLine={false} tickMargin={6} minTickGap={16} />
                <YAxis width={40} tickLine={false} axisLine={false} />
                <ChartTooltip
                  cursor={{ fill: "var(--muted)", opacity: 0.5 }}
                  content={
                    <ChartTooltipContent
                      hideIndicator
                      labelFormatter={(_, p) => {
                        const r = p?.[0]?.payload
                        return r ? `${fmt.clock(r.time)} · ${fmt.n(r.gen_tokens)} tokens · ${r.api}` : ""
                      }}
                      formatter={(v) => <span className="font-mono tabular">{fmt.n(Number(v), 1)} tok/s</span>}
                    />
                  }
                />
                <Bar dataKey="tps" fill="var(--color-tps)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ChartContainer>
          ) : (
            <NoData />
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Prompt speed by prompt size</CardTitle>
          <CardDescription>Longer prompts amortize the expert transfers over more tokens</CardDescription>
        </CardHeader>
        <CardContent>
          {reqs.length ? (
            <ChartContainer config={{ pf: { label: "Prefill", color: "var(--chart-2)" } }} className="aspect-auto h-[220px] w-full">
              <ScatterChart margin={{ top: 8, right: 8 }}>
                <CartesianGrid strokeOpacity={0.5} />
                <XAxis dataKey="fresh" type="number" name="New prompt tokens" domain={[0, "auto"]}
                  tickFormatter={(v) => fmt.compact(v)} tickLine={false} axisLine={false} />
                <YAxis dataKey="prefill_tps" type="number" width={44} domain={[0, "auto"]} tickLine={false} axisLine={false} tickFormatter={(v) => fmt.compact(v)} />
                <ZAxis range={[64, 64]} />
                <ChartTooltip
                  cursor={false}
                  content={
                    <ChartTooltipContent
                      hideIndicator
                      labelFormatter={() => "Request"}
                      formatter={(_, __, item) => (
                        <div className="grid gap-0.5 font-mono tabular">
                          <span>{fmt.n(item.payload.fresh)} new tokens</span>
                          <span>{fmt.n(item.payload.prefill_tps)} tok/s</span>
                        </div>
                      )}
                    />
                  }
                />
                <Scatter
                  data={reqs.filter((r) => (r.prefill_tps ?? 0) > 0).map((r) => ({ ...r, fresh: Math.max(1, r.prompt_tokens - (r.reused ?? 0)) }))}
                  fill="var(--color-pf)"
                  stroke="var(--card)"
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              </ScatterChart>
            </ChartContainer>
          ) : (
            <NoData />
          )}
        </CardContent>
      </Card>

      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>Where a decode round goes</CardTitle>
          <CardDescription>Averages over every round since the engine started</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-6 sm:grid-cols-4">
          <Figure label="Verify pass" value={fmt.ms(perRound(life?.verify_ms))} note="Full model over the drafted window" />
          <Figure label="CPU experts" value={fmt.ms(perRound(life?.cpu_expert_ms))} note="Inside the verify pass, overlapped with the GPU" />
          <Figure label="Drafting" value={fmt.ms(perRound(life?.draft_ms))} note="MTP layer proposing the next tokens" />
          <Figure label="Rounds" value={fmt.compact(life?.rounds)} note={`${fmt.n(life && life.rounds ? life.gen_tokens / life.rounds : null, 2)} tokens each`} />
        </CardContent>
      </Card>
    </div>
  )
}

function Figure({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="grid content-start gap-1">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-2xl font-semibold tabular">{value}</span>
      <span className="text-xs text-muted-foreground">{note}</span>
    </div>
  )
}

export function NoData({ text = "No requests yet" }: { text?: string }) {
  return <div className="flex h-[220px] items-center justify-center text-sm text-muted-foreground">{text}</div>
}
