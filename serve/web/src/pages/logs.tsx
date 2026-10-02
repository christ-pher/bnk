import { useEffect, useRef, useState } from "react"

import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { useTelemetry } from "@/lib/telemetry"

export function LogsPage() {
  const log = useTelemetry((s) => s.log)
  const args = useTelemetry((s) => s.overview?.engine.args)
  const [filter, setFilter] = useState("")
  const [follow, setFollow] = useState(true)
  const end = useRef<HTMLDivElement>(null)
  const lines = filter ? log.filter((l) => l.line.toLowerCase().includes(filter.toLowerCase())) : log
  useEffect(() => {
    if (follow) end.current?.scrollIntoView({ block: "end" })
  }, [lines.length, follow])

  return (
    <div className="grid gap-4 md:gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Engine log</CardTitle>
          <CardDescription>The engine's stderr, streamed live</CardDescription>
          <CardAction className="flex items-center gap-4">
            <Input placeholder="Filter…" value={filter} onChange={(e) => setFilter(e.target.value)} className="h-8 w-40 sm:w-56" />
            <div className="flex items-center gap-2">
              <Switch id="follow" checked={follow} onCheckedChange={setFollow} />
              <Label htmlFor="follow" className="text-sm">Follow</Label>
            </div>
          </CardAction>
        </CardHeader>
        <CardContent>
          <div className="h-[60vh] overflow-auto rounded-lg border bg-muted/30 p-3 font-mono text-xs leading-relaxed">
            {lines.map((l, i) => (
              <div key={i} className={l.line.match(/error|fail|exception/i) ? "text-critical" : undefined}>{l.line}</div>
            ))}
            <div ref={end} />
          </div>
        </CardContent>
      </Card>
      {args && (
        <Card>
          <CardHeader>
            <CardTitle>Engine command line</CardTitle>
          </CardHeader>
          <CardContent>
            <code className="block rounded-lg border bg-muted/30 p-3 font-mono text-xs break-all">bnk serve {args.join(" ")}</code>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
