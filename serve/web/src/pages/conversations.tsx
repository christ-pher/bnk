import { ChevronRight, Cpu, HardDrive, MemoryStick, MessagesSquare, Timer, Zap } from "lucide-react"
import { Fragment, useState, type ReactNode } from "react"

import { StatCard } from "@/components/dash/stat-card"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Progress } from "@/components/ui/progress"
import { Spinner } from "@/components/ui/spinner"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { fmt } from "@/lib/format"
import { useNow, useTelemetry, type Conversation, type ConvTurn, type Live } from "@/lib/telemetry"

const ACTIVE_S = 15 * 60 // "active" conversations: a turn within the last 15 minutes

// Several agents (or chats) share the engine: requests run one at a time, and each conversation's state is parked
// in host RAM between its turns, so a turn reads only its new tokens.
export function ConversationsPage() {
  const convs = useTelemetry((s) => s.conversations)
  const live = useTelemetry((s) => s.live)
  const now = useNow() / 1000
  const [open, setOpen] = useState<Set<number>>(new Set())
  const toggle = (id: number) =>
    setOpen((s) => {
      const n = new Set(s)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })

  const active = convs.filter((c) => c.status !== "idle" || now - c.last_active < ACTIVE_S)
  const running = convs.find((c) => c.status === "reading" || c.status === "generating")
  const queued = convs.filter((c) => c.status === "queued")
  // turns after a conversation's first one: what parking is for
  const later = convs.flatMap((c) => c.turns.filter((t) => (t.reused ?? 0) > 0)).sort((a, b) => b.time - a.time).slice(0, 30)
  const avgRead = later.length ? later.reduce((s, t) => s + (t.prefill_ms ?? 0), 0) / later.length : null
  const reusedShare = later.length
    ? later.reduce((s, t) => s + (t.reused ?? 0), 0) / Math.max(1, later.reduce((s, t) => s + t.prompt_tokens, 0))
    : null
  const life = live?.life

  return (
    <div className="flex min-h-full flex-col gap-4 md:gap-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Active conversations"
          value={fmt.n(active.length)}
          action={<MessagesSquare className="size-4 text-muted-foreground" />}
          footer={<span>With a turn in the last 15 minutes · {fmt.n(convs.length)} tracked</span>}
        />
        <StatCard
          label="Now"
          value={running ? `#${running.id}` : "Idle"}
          unit={running ? (running.status === "reading" ? "reading" : "generating") : undefined}
          action={running ? <Spinner className="size-4" /> : <Zap className="size-4 text-muted-foreground" />}
          footer={<span>{queued.length ? `${queued.length} waiting: ${queued.map((c) => `#${c.id}`).join(", ")}` : "Nothing waiting"}</span>}
        />
        <StatCard
          label="Parked in RAM"
          value={fmt.n(live?.parked ?? 0)}
          unit={live?.parked_gb ? `${fmt.n(live.parked_gb, 2)} GiB` : undefined}
          action={<HardDrive className="size-4 text-muted-foreground" />}
          footer={
            <span>
              {life?.parks ? (
                <>
                  {fmt.n(life.parks)} out ({fmt.ms((life.park_ms ?? 0) / life.parks)}) · {fmt.n(life.restores)} back (
                  {fmt.ms((life.restore_ms ?? 0) / Math.max(1, life.restores ?? 0))})
                  {life.park_evictions ? ` · ${fmt.n(life.park_evictions)} dropped` : ""}
                </>
              ) : (
                "No conversation parked yet"
              )}
            </span>
          }
        />
        <StatCard
          label="Turn reads"
          value={avgRead == null ? "—" : fmt.ms(avgRead)}
          unit={avgRead == null ? undefined : "avg"}
          action={<Timer className="size-4 text-muted-foreground" />}
          footer={<span>{reusedShare == null ? "After a conversation's first turn" : `Last ${later.length} follow-up turns · ${fmt.pct(reusedShare, 1)} of their prompts reused`}</span>}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Conversations</CardTitle>
          <CardDescription>
            Requests that continue each other's tokens, most recent first. One runs at a time; the others wait in line or
            sit parked in host RAM until their next turn. Click a row for its turns.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {convs.length === 0 ? (
            <Empty>
              <EmptyHeader>
                <EmptyTitle>No conversations yet</EmptyTitle>
                <EmptyDescription>They appear here as requests arrive through the API or the test chat.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <Table className="tabular">
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8" />
                  <TableHead>Conversation</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Context</TableHead>
                  <TableHead>State</TableHead>
                  <TableHead className="text-right">Turns</TableHead>
                  <TableHead className="text-right">Last read</TableHead>
                  <TableHead className="text-right">Last output</TableHead>
                  <TableHead className="text-right">Generated</TableHead>
                  <TableHead className="text-right">Last active</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {convs.map((c) => {
                  const last = c.turns.at(-1)
                  const isOpen = open.has(c.id)
                  return (
                    <Fragment key={c.id}>
                      <TableRow className="cursor-pointer" onClick={() => toggle(c.id)} data-state={c.status !== "idle" ? "selected" : undefined}>
                        <TableCell>
                          <ChevronRight className={`size-4 text-muted-foreground transition-transform ${isOpen ? "rotate-90" : ""}`} />
                        </TableCell>
                        <TableCell className="max-w-[22rem]">
                          <div className="flex items-baseline gap-2">
                            <span className="font-medium">#{c.id}</span>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="truncate text-muted-foreground">{c.label}</span>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-md">{c.label}</TooltipContent>
                            </Tooltip>
                          </div>
                        </TableCell>
                        <TableCell>
                          <StatusCell c={c} live={live} now={now} />
                        </TableCell>
                        <TableCell className="text-right">{fmt.n(c.context)}</TableCell>
                        <TableCell>
                          <WhereBadge where={c.where} />
                        </TableCell>
                        <TableCell className="text-right">{fmt.n(c.n_turns)}</TableCell>
                        <TableCell className="text-right">{last ? <ReadCell t={last} /> : "—"}</TableCell>
                        <TableCell className="text-right">
                          {last ? (
                            <span>
                              {fmt.n(last.gen_tokens)} <span className="text-muted-foreground">· {fmt.n(last.tps, 1)} tok/s</span>
                              {(last.slices ?? 1) > 1 && <SlicesBadge n={last.slices!} />}
                            </span>
                          ) : (
                            "—"
                          )}
                        </TableCell>
                        <TableCell className="text-right">{fmt.n(c.gen_total)}</TableCell>
                        <TableCell className="text-right text-muted-foreground">{agoAt(c.last_active, now)}</TableCell>
                      </TableRow>
                      {isOpen && (
                        <TableRow className="hover:bg-transparent">
                          <TableCell />
                          <TableCell colSpan={9} className="pb-4">
                            <Turns c={c} />
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
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

function StatusCell({ c, live, now }: { c: Conversation; live: Live | null; now: number }) {
  const r = live?.req && live.req.id === c.rid ? live.req : null
  if (c.status === "generating")
    return (
      <div className="flex items-center gap-2">
        <Badge className="gap-1.5"><Zap className="size-3" />Generating</Badge>
        {r && (
          <span className="text-xs text-muted-foreground">
            {fmt.n(r.gen_tokens)} tok{r.gen_ms > 400 ? ` · ${fmt.n((r.gen_tokens - 1) / (r.gen_ms / 1000), 1)} tok/s` : ""}
          </span>
        )}
      </div>
    )
  if (c.status === "reading")
    return (
      <div className="flex min-w-36 items-center gap-2">
        <Badge className="gap-1.5"><Spinner className="size-3" />Reading</Badge>
        {r && r.prefill_total > 0 && <Progress className="h-1.5 w-16" value={(100 * r.prefill_done) / r.prefill_total} />}
      </div>
    )
  if (c.status === "queued")
    return (
      <Badge variant="outline" className="gap-1.5 border-warning/60">
        <span className="size-1.5 rounded-full bg-warning" />
        Waiting {fmt.duration(Math.max(0, now - c.since))}
      </Badge>
    )
  return <Badge variant="secondary">Idle</Badge>
}

function WhereBadge({ where }: { where: Conversation["where"] }) {
  const v: Record<Conversation["where"], { icon: ReactNode; text: string; tip: string }> = {
    gpu: { icon: <Cpu className="size-3" />, text: "GPU", tip: "The engine holds this conversation now: its next turn continues at once" },
    ram: { icon: <MemoryStick className="size-3" />, text: "RAM", tip: "Parked in host RAM: its next turn brings it back in a fraction of a second" },
    none: { icon: null, text: "—", tip: "Too short to park (under 2,048 tokens) or parking is off: its next turn reads it again" },
  }
  const x = v[where]
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="outline" className="gap-1">{x.icon}{x.text}</Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{x.tip}</TooltipContent>
    </Tooltip>
  )
}

function ReadCell({ t }: { t: ConvTurn }) {
  const reused = t.reused ?? 0
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span>
          {fmt.ms(t.prefill_ms)}{" "}
          <span className="text-muted-foreground">· {reused ? fmt.pct(reused / Math.max(1, t.prompt_tokens)) : "new"}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent>
        {fmt.n(t.prompt_tokens - reused)} new of {fmt.n(t.prompt_tokens)} prompt tokens ({fmt.n(reused)} reused)
      </TooltipContent>
    </Tooltip>
  )
}

function SlicesBadge({ n }: { n: number }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="outline" className="ml-1.5 px-1.5">{n}×</Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">Ran in {n} time slices: it gave its turn to waiting requests in between</TooltipContent>
    </Tooltip>
  )
}

function Turns({ c }: { c: Conversation }) {
  const rows = [...c.turns].reverse()
  if (!rows.length) return <p className="text-sm text-muted-foreground">No finished turns yet.</p>
  return (
    <div className="rounded-md border">
      <Table className="tabular text-xs">
        <TableHeader>
          <TableRow>
            <TableHead>Time</TableHead>
            <TableHead className="text-right">Prompt</TableHead>
            <TableHead className="text-right">New</TableHead>
            <TableHead className="text-right">Read</TableHead>
            <TableHead className="text-right">Output</TableHead>
            <TableHead className="text-right">Decode</TableHead>
            <TableHead className="text-right">Slices</TableHead>
            <TableHead className="text-right">Wall time</TableHead>
            <TableHead>Finish</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((t) => (
            <TableRow key={t.id}>
              <TableCell className="text-muted-foreground">{fmt.clock(t.time)}</TableCell>
              <TableCell className="text-right">{fmt.n(t.prompt_tokens)}</TableCell>
              <TableCell className="text-right">{fmt.n(t.prompt_tokens - (t.reused ?? 0))}</TableCell>
              <TableCell className="text-right">{fmt.ms(t.prefill_ms)}</TableCell>
              <TableCell className="text-right">{fmt.n(t.gen_tokens)}</TableCell>
              <TableCell className="text-right">{fmt.n(t.tps, 1)} tok/s</TableCell>
              <TableCell className="text-right">{fmt.n(t.slices ?? 1)}</TableCell>
              <TableCell className="text-right text-muted-foreground">{t.wall_s != null ? fmt.ms(t.wall_s * 1000) : "—"}</TableCell>
              <TableCell><Badge variant={t.finish === "stop" ? "secondary" : "outline"}>{t.finish}</Badge></TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function agoAt(t: number, now: number) {
  const s = now - t
  if (s < 5) return "just now"
  if (s < 60) return `${Math.floor(s)}s ago`
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  return `${Math.floor(s / 3600)}h ago`
}
