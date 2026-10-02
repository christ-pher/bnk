import { useState } from "react"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"

import { TimeChart } from "@/components/dash/time-chart"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { fmt } from "@/lib/format"
import { layerWindow, RANGES, useTelemetry, windowed, type Range } from "@/lib/telemetry"

import { NoData } from "./performance"

export function ExpertsPage({ range }: { range: Range }) {
  const live = useTelemetry((s) => s.live)
  const history = useTelemetry((s) => s.history)
  const [win, setWin] = useState<"range" | "life">("range")
  const rows = windowed(history, range)
  if (!live?.layer_slots) return <NoData text="Waiting for the engine…" />

  const w = layerWindow(live, win === "life" ? null : RANGES[range])
  const layers = live.layer_slots.map((slots, il) => ({
    layer: il,
    resident: slots / live.n_expert,
    slots,
    routed: w.routed[il],
    misses: w.misses[il],
    miss: w.routed[il] > 0 ? w.misses[il] / w.routed[il] : null,
  }))
  const totalR = w.routed.reduce((a, b) => a + b, 0)
  const totalM = w.misses.reduce((a, b) => a + b, 0)
  const worst = [...layers].filter((l) => l.miss != null).sort((a, b) => (b.miss ?? 0) - (a.miss ?? 0)).slice(0, 5)

  return (
    <div className="grid grid-cols-1 gap-4 md:gap-6 xl:grid-cols-3">
      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>CPU misses by layer</CardTitle>
          <CardDescription>
            Share of expert uses not resident in VRAM (served by the CPU) ·{" "}
            {totalR ? `${fmt.pct(totalM / totalR, 1)} overall, ${fmt.compact(totalR)} uses` : "no traffic in this window"}
          </CardDescription>
          <CardAction>
            <ToggleGroup type="single" variant="outline" size="sm" value={win} onValueChange={(v) => v && setWin(v as "range" | "life")}>
              <ToggleGroupItem value="range">Last {range}</ToggleGroupItem>
              <ToggleGroupItem value="life">Lifetime</ToggleGroupItem>
            </ToggleGroup>
          </CardAction>
        </CardHeader>
        <CardContent>
          {totalR ? (
            <ChartContainer config={{ miss: { label: "Miss rate", color: "var(--chart-1)" } }} className="aspect-auto h-[260px] w-full">
              <BarChart data={layers} margin={{ top: 8, right: 8 }} barCategoryGap={2}>
                <CartesianGrid vertical={false} strokeOpacity={0.5} />
                <XAxis dataKey="layer" tickLine={false} axisLine={false} interval={3} tickMargin={6} />
                <YAxis width={40} tickLine={false} axisLine={false} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <ChartTooltip
                  cursor={{ fill: "var(--muted)", opacity: 0.5 }}
                  content={
                    <ChartTooltipContent
                      hideIndicator
                      labelFormatter={(_, p) => `Layer ${p?.[0]?.payload?.layer}`}
                      formatter={(_, __, item) => (
                        <div className="grid gap-0.5 font-mono tabular">
                          <span>{fmt.pct(item.payload.miss, 1)} missed</span>
                          <span className="text-muted-foreground">{fmt.n(item.payload.misses)} of {fmt.n(item.payload.routed)} uses</span>
                          <span className="text-muted-foreground">{item.payload.slots} of {live.n_expert} experts resident</span>
                        </div>
                      )}
                    />
                  }
                />
                <Bar dataKey="miss" fill="var(--color-miss)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ChartContainer>
          ) : (
            <NoData text="No expert traffic in this window" />
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Hottest layers</CardTitle>
          <CardDescription>Most CPU misses in the selected window</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          {worst.length ? (
            worst.map((l) => (
              <div key={l.layer} className="grid gap-1.5">
                <div className="flex justify-between text-sm">
                  <span>Layer {l.layer}</span>
                  <span className="font-medium tabular">{fmt.pct(l.miss, 1)}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-chart-1" style={{ width: `${(l.miss ?? 0) * 100}%` }} />
                </div>
                <span className="text-xs text-muted-foreground">{l.slots} of {live.n_expert} experts resident · {fmt.compact(l.routed)} uses</span>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">No traffic yet.</p>
          )}
        </CardContent>
      </Card>

      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>VRAM residency by layer</CardTitle>
          <CardDescription>Cache slots per layer, sized from the routing statistics ({fmt.n(live.expert_cache_gb, 1)} GiB in all)</CardDescription>
          <CardAction>
            <Badge variant="outline" className="tabular">{fmt.n(live.experts_resident)} / {fmt.n(live.experts_total)}</Badge>
          </CardAction>
        </CardHeader>
        <CardContent>
          <ChartContainer config={{ resident: { label: "Resident", color: "var(--chart-1)" } }} className="aspect-auto h-[200px] w-full">
            <BarChart data={layers} margin={{ top: 8, right: 8 }} barCategoryGap={2}>
              <CartesianGrid vertical={false} strokeOpacity={0.5} />
              <XAxis dataKey="layer" tickLine={false} axisLine={false} interval={3} tickMargin={6} />
              <YAxis width={40} tickLine={false} axisLine={false} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <ChartTooltip
                cursor={{ fill: "var(--muted)", opacity: 0.5 }}
                content={
                  <ChartTooltipContent
                    hideIndicator
                    labelFormatter={(_, p) => `Layer ${p?.[0]?.payload?.layer}`}
                    formatter={(_, __, item) => (
                      <span className="font-mono tabular">{item.payload.slots} of {live.n_expert} experts ({fmt.pct(item.payload.resident)})</span>
                    )}
                  />
                }
              />
              <Bar dataKey="resident" fill="var(--color-resident)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Cache adaptation</CardTitle>
          <CardDescription>Experts swapped into VRAM per second</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} series={[{ key: "sw", label: "Swaps", color: "var(--chart-1)", value: (r) => r.swaps }]} unit="/s" height={200} />
        </CardContent>
      </Card>

      <Card className="xl:col-span-3">
        <CardHeader>
          <CardTitle>VRAM hit rate</CardTitle>
          <CardDescription>Share of expert uses served from the VRAM cache, per second of traffic</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} kind="line" series={[{ key: "hit", label: "Hit rate", color: "var(--chart-1)", value: (r) => (r.miss_rate == null ? null : 100 * (1 - r.miss_rate)), missing: "connect" }]} unit="%" digits={1} domain={[0, 100]} />
        </CardContent>
      </Card>
    </div>
  )
}
