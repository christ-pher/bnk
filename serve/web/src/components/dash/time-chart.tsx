// A time series over the shared 1 s history: thin 2px lines over a faint area, hairline grid, crosshair tooltip.
import { Area, AreaChart, CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart"
import { fmt } from "@/lib/format"

export interface Series {
  key: string
  label: string
  color: string // a --chart-n token
  value: (row: any) => number | null | undefined // a missing reading (no work of that kind that second) plots as 0
}

export function TimeChart({
  rows,
  series,
  unit = "",
  digits = 0,
  domain,
  kind = "area",
  height = 200,
}: {
  rows: { t: number }[]
  series: Series[]
  unit?: string
  digits?: number
  domain?: [number | "auto" | "dataMin" | "dataMax", number | "auto" | "dataMin" | "dataMax"]
  kind?: "area" | "line"
  height?: number | string // a number of px, or "100%" to fill a sized parent
}) {
  const config: ChartConfig = Object.fromEntries(series.map((s) => [s.key, { label: s.label, color: s.color }]))
  if (rows.length < 2)
    return (
      <div className="flex w-full items-center justify-center text-sm text-muted-foreground" style={{ height }}>
        Collecting samples…
      </div>
    )
  const data = rows.map((r) => {
    const o: Record<string, number | null> = { t: r.t }
    for (const s of series) {
      const v = s.value(r)
      o[s.key] = v == null || Number.isNaN(v) ? 0 : Number(v.toFixed(digits))
    }
    return o
  })
  const tooltip = (
    <ChartTooltip
      cursor={{ stroke: "var(--border)", strokeWidth: 1 }}
      isAnimationActive={false}
      content={
        <ChartTooltipContent
          indicator="line"
          labelFormatter={(_, p) => (p?.[0]?.payload?.t ? fmt.clock(p[0].payload.t) : "")}
          formatter={(value, name, item) => (
            <div className="flex w-full items-center gap-2">
              <div className="h-2.5 w-1 shrink-0 rounded-[2px]" style={{ background: item.color }} />
              <span className="text-muted-foreground">{config[String(name)]?.label ?? name}</span>
              <span className="ml-auto font-mono font-medium text-foreground tabular-nums">
                {fmt.n(Number(value), digits)}
                {unit && <span className="ml-0.5 text-muted-foreground">{unit}</span>}
              </span>
            </div>
          )}
        />
      }
    />
  )
  const axes = (
    <>
      <CartesianGrid vertical={false} strokeOpacity={0.5} />
      {/* time runs left to right; the moving clock labels are left out (the tooltip shows the time) */}
      <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} hide />
      <YAxis
        width={44}
        tickLine={false}
        axisLine={false}
        tickMargin={4}
        domain={domain ?? [0, "auto"]}
        tickFormatter={(v) => fmt.compact(v)}
        className="tabular"
      />
      {tooltip}
      {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
    </>
  )
  return (
    <ChartContainer config={config} className="aspect-auto w-full" style={{ height }}>
      {kind === "area" ? (
        <AreaChart data={data} margin={{ left: 0, right: 8, top: 8 }}>
          <defs>
            {series.map((s) => (
              <linearGradient key={s.key} id={`fill-${s.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={`var(--color-${s.key})`} stopOpacity={0.28} />
                <stop offset="95%" stopColor={`var(--color-${s.key})`} stopOpacity={0.02} />
              </linearGradient>
            ))}
          </defs>
          {axes}
          {series.map((s) => (
            <Area
              key={s.key}
              dataKey={s.key}
              type="monotone"
              stroke={`var(--color-${s.key})`}
              strokeWidth={2}
              fill={`url(#fill-${s.key})`}
              isAnimationActive={false}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--card)" }}
            />
          ))}
        </AreaChart>
      ) : (
        <LineChart data={data} margin={{ left: 0, right: 8, top: 8 }}>
          {axes}
          {series.map((s) => (
            <Line
              key={s.key}
              dataKey={s.key}
              type="monotone"
              stroke={`var(--color-${s.key})`}
              strokeWidth={2}
              isAnimationActive={false}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--card)" }}
            />
          ))}
        </LineChart>
      )}
    </ChartContainer>
  )
}

// A bare trend line for stat tiles: no axes, no tooltip (the tile's number is the reading).
export function Sparkline({ values, color = "var(--chart-1)", height = 36 }: { values: (number | null)[]; color?: string; height?: number }) {
  const data = values.map((v, i) => ({ i, v: v == null || Number.isNaN(v) ? 0 : v }))
  return (
    <ChartContainer config={{ v: { label: "", color } }} className="aspect-auto w-full" style={{ height }}>
      <AreaChart data={data} margin={{ top: 2, bottom: 0, left: 0, right: 0 }}>
        <defs>
          <linearGradient id={`spark-${color.replace(/[^a-z0-9]/gi, "")}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.3} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <YAxis hide domain={[0, "auto"]} />
        <Area
          dataKey="v"
          type="monotone"
          stroke={color}
          strokeWidth={1.5}
          fill={`url(#spark-${color.replace(/[^a-z0-9]/gi, "")})`}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ChartContainer>
  )
}
