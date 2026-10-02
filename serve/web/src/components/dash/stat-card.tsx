import type { ReactNode } from "react"

import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export function StatCard({
  label,
  value,
  unit,
  action,
  footer,
  children,
}: {
  label: string
  value: ReactNode
  unit?: string
  action?: ReactNode
  footer?: ReactNode
  children?: ReactNode
}) {
  return (
    <Card className="@container/card gap-3">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="flex items-baseline gap-1.5 text-3xl font-semibold tracking-tight @[250px]/card:text-4xl">
          {value}
          {unit && <span className="text-sm font-normal text-muted-foreground">{unit}</span>}
        </CardTitle>
        {action && <CardAction>{action}</CardAction>}
      </CardHeader>
      {children && <CardContent>{children}</CardContent>}
      {footer && <CardContent className="mt-auto flex flex-col gap-0.5 text-xs text-muted-foreground">{footer}</CardContent>}
    </Card>
  )
}

// A labelled meter row: name, reading, and a thin bar.
export function Meter({ label, reading, value, hint }: { label: string; reading: ReactNode; value: number | null; hint?: ReactNode }) {
  const v = value == null || Number.isNaN(value) ? 0 : Math.max(0, Math.min(1, value))
  return (
    <div className="grid gap-1.5">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular">{reading}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-chart-1 transition-[width] duration-500" style={{ width: `${v * 100}%` }} />
      </div>
      {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
    </div>
  )
}
