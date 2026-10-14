import { TimeChart, type Series } from "@/components/dash/time-chart"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { fmt } from "@/lib/format"
import { useTelemetry, windowed, type GpuSample, type Range } from "@/lib/telemetry"

const GPU_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"]

export function SystemPage({ range }: { range: Range }) {
  const history = useTelemetry((s) => s.history)
  const live = useTelemetry((s) => s.live)
  const gpu = useTelemetry((s) => s.overview?.gpu)
  const rows = windowed(history, range)
  const cores = live?.cores ?? []
  const ramGiB = (live?.ram_total_mb ?? 0) / 1024
  // several GPUs: a line per GPU (GPU i keeps its color on every chart), and each one's reading in the badges
  const nGpu = rows.at(-1)?.gpus?.length ?? 0
  const multi = nGpu > 1 // charts: samples that hold each GPU's readings
  const multiLive = (live?.gpu?.gpus?.length ?? 0) > 1 // badges
  const perGpu = (key: string, f: (g: GpuSample) => number | null | undefined, suffix = ""): Series[] =>
    Array.from({ length: nGpu }, (_, i) => ({
      key: `${key}${i}`,
      label: `GPU ${i}${suffix}`,
      color: GPU_COLORS[i % GPU_COLORS.length],
      value: (r) => {
        const g = r.gpus?.[i]
        return g ? f(g) : null
      },
    }))
  const each = (f: (g: NonNullable<NonNullable<typeof live>["gpu"]>) => number | null | undefined, unit: string) =>
    (live?.gpu?.gpus ?? []).map((g) => `${fmt.n(f(g))}${unit}`).join(" · ")
  const gpuLimit = Math.max(0, ...(gpu?.gpus ?? []).map((g) => g.power_limit_w ?? 0)) || gpu?.power_limit_w

  return (
    <div className="grid grid-cols-1 gap-4 md:gap-6 lg:grid-cols-2 xl:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>GPU utilization</CardTitle>
          <CardDescription>{gpu?.name ?? "GPU"}</CardDescription>
          <CardAction><Badge variant="outline" className="tabular">{multiLive ? each((g) => g.util, "%") : `${fmt.n(live?.gpu?.util)}%`}</Badge></CardAction>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} series={multi ? perGpu("u", (g) => g.util) : [{ key: "u", label: "Utilization", color: "var(--chart-1)", value: (r) => r.gpu_util }]} unit="%" domain={[0, 100]} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>GPU power</CardTitle>
          <CardDescription>
            Board draw{multiLive && gpuLimit ? ` · limit ${fmt.n(gpuLimit)} W each` : gpu?.power_limit_w ? ` · limit ${fmt.n(gpu.power_limit_w)} W` : ""}
          </CardDescription>
          <CardAction><Badge variant="outline" className="tabular">{multiLive ? each((g) => g.power_w, " W") : `${fmt.n(live?.gpu?.power_w)} W`}</Badge></CardAction>
        </CardHeader>
        <CardContent>
          <TimeChart
            rows={rows}
            series={multi ? perGpu("p", (g) => g.power_w) : [{ key: "p", label: "Power", color: "var(--chart-2)", value: (r) => r.gpu_power_w }]}
            unit="W"
            domain={[0, (multi ? gpuLimit : gpu?.power_limit_w) ?? "auto"]}
          />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>PCIe traffic</CardTitle>
          <CardDescription>Gen {gpu?.pcie_gen ?? "—"} x{gpu?.pcie_width ?? "—"} · expert transfers</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart
            rows={rows}
            series={multi ? [...perGpu("rx", (g) => (g.pcie_rx_mbs == null ? null : g.pcie_rx_mbs / 1024), " ← host"), ...perGpu("tx", (g) => (g.pcie_tx_mbs == null ? null : g.pcie_tx_mbs / 1024), " → host").map((x, i) => ({ ...x, color: GPU_COLORS[(nGpu + i) % GPU_COLORS.length] }))] : [
              { key: "rx", label: "Host → GPU", color: "var(--chart-1)", value: (r) => (r.pcie_rx_mbs == null ? null : r.pcie_rx_mbs / 1024) },
              { key: "tx", label: "GPU → host", color: "var(--chart-2)", value: (r) => (r.pcie_tx_mbs == null ? null : r.pcie_tx_mbs / 1024) },
            ]}
            unit="GB/s"
            digits={2}
          />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>GPU temperature</CardTitle>
          <CardDescription>SM clock {multiLive ? each((g) => g.sm_clock, "") : fmt.n(live?.gpu?.sm_clock)} MHz</CardDescription>
          <CardAction><Badge variant="outline" className="tabular">{multiLive ? each((g) => g.temp_c, " °C") : `${fmt.n(live?.gpu?.temp_c)} °C`}</Badge></CardAction>
        </CardHeader>
        <CardContent>
          <TimeChart rows={rows} kind="line" series={multi ? perGpu("tc", (g) => g.temp_c) : [{ key: "tc", label: "Temperature", color: "var(--chart-2)", value: (r) => r.gpu_temp_c }]} unit="°C" domain={[20, 90]} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Memory</CardTitle>
          <CardDescription>VRAM {fmt.gib(live?.vram_total_mb)} · RAM {fmt.n(ramGiB, 0)} GiB</CardDescription>
        </CardHeader>
        <CardContent>
          <TimeChart
            rows={rows}
            kind="line"
            series={[
              ...(multi
                ? perGpu("vram", (g) => (g.vram_used_mb == null ? null : g.vram_used_mb / 1024), " VRAM")
                : [{ key: "vram", label: "VRAM used", color: "var(--chart-1)", value: (r: any) => r.vram_used_mb / 1024 }]),
              { key: "ram", label: "RAM used", color: GPU_COLORS[(multi ? nGpu : 1) % GPU_COLORS.length], value: (r) => r.ram_used_mb / 1024 },
            ]}
            unit="GiB"
            digits={1}
          />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>CPU</CardTitle>
          <CardDescription>{cores.length || "—"} cores · {live?.cpu_threads ?? "—"} expert threads</CardDescription>
          <CardAction><Badge variant="outline" className="tabular">{fmt.n(history.at(-1)?.cpu)}%</Badge></CardAction>
        </CardHeader>
        <CardContent className="grid gap-4">
          <TimeChart rows={rows} series={[{ key: "cpu", label: "CPU", color: "var(--chart-1)", value: (r) => r.cpu }]} unit="%" domain={[0, 100]} height={150} />
          <div className="grid grid-cols-8 gap-1 sm:grid-cols-16">
            {cores.map((u, i) => (
              <Tooltip key={i}>
                <TooltipTrigger asChild>
                  <div
                    className="flex aspect-square items-center justify-center rounded-sm text-[10px] font-medium tabular"
                    style={{
                      background: `color-mix(in oklab, var(--chart-1) ${Math.round(Math.min(100, u))}%, var(--muted))`,
                      color: u > 55 ? "white" : "var(--muted-foreground)",
                    }}
                  >
                    {Math.round(u)}
                  </div>
                </TooltipTrigger>
                <TooltipContent>Core {i} · {fmt.n(u)}% busy</TooltipContent>
              </Tooltip>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
