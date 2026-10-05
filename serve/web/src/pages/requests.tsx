import { RefreshCcw } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { fmt } from "@/lib/format"
import { useTelemetry } from "@/lib/telemetry"

export function RequestsPage() {
  const recent = useTelemetry((s) => s.recent)
  const rows = [...recent].reverse()
  return (
    <Card>
      <CardHeader>
        <CardTitle>Requests</CardTitle>
        <CardDescription>The last {rows.length} requests through any API, newest first</CardDescription>
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>No requests yet</EmptyTitle>
              <EmptyDescription>Send one through the OpenAI or Anthropic API, or open the test chat.</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <Table className="tabular">
            <TableHeader>
              <TableRow>
                <TableHead>Time</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Conv.</TableHead>
                <TableHead className="text-right">Prompt</TableHead>
                <TableHead className="text-right">Reused</TableHead>
                <TableHead className="text-right">Prefill</TableHead>
                <TableHead className="text-right">Output</TableHead>
                <TableHead className="text-right">Decode</TableHead>
                <TableHead className="text-right">Tok / round</TableHead>
                <TableHead className="text-right">Accepted</TableHead>
                <TableHead className="text-right">CPU misses</TableHead>
                <TableHead className="text-right">Temp</TableHead>
                <TableHead>Finish</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => (
                <TableRow key={r.id ?? i}>
                  <TableCell className="text-muted-foreground">{fmt.clock(r.time)}</TableCell>
                  <TableCell><Badge variant="outline">{r.api || "—"}</Badge></TableCell>
                  <TableCell>{r.conversation != null ? <a href="#conversations" className="hover:underline">#{r.conversation}</a> : "—"}</TableCell>
                  <TableCell className="text-right">{fmt.n(r.prompt_tokens)}</TableCell>
                  <TableCell className="text-right text-muted-foreground">{fmt.n(r.reused)}</TableCell>
                  <TableCell className="text-right">{r.prefill_tps ? `${fmt.n(r.prefill_tps)} tok/s` : fmt.ms(r.prefill_ms)}</TableCell>
                  <TableCell className="text-right">{fmt.n(r.gen_tokens)}</TableCell>
                  <TableCell className="text-right font-medium">{fmt.n(r.tps, 1)} tok/s</TableCell>
                  <TableCell className="text-right">{fmt.n(r.tokens_per_round, 2)}</TableCell>
                  <TableCell className="text-right">{r.drafted ? fmt.pct((r.accepted ?? 0) / r.drafted) : "—"}</TableCell>
                  <TableCell className="text-right">{fmt.pct(r.expert_miss_rate, 1)}</TableCell>
                  <TableCell className="text-right text-muted-foreground">{fmt.n(r.temperature, 1)}</TableCell>
                  <TableCell className="flex items-center gap-1.5">
                    <Badge variant={r.finish === "stop" ? "secondary" : "outline"}>{r.finish}</Badge>
                    {r.loop_guard ? (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Badge variant="outline" className="gap-1 border-warning/60">
                            <RefreshCcw className="size-3" />
                            loop
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent>The reasoning kept repeating itself; the thinking-loop guard closed it so the model could answer</TooltipContent>
                      </Tooltip>
                    ) : null}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
