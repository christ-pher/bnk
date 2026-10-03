import { Cpu, Gauge, LayoutDashboard, ListOrdered, MessageSquare, Moon, ScrollText, Sun, Zap } from "lucide-react"
import { useEffect, useState } from "react"

import { ChatSheet, type ChatSettings, type Msg } from "@/components/chat/chat-sheet"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { TooltipProvider } from "@/components/ui/tooltip"
import { fmt } from "@/lib/format"
import { RANGES, useTelemetry, type Range } from "@/lib/telemetry"
import { ExpertsPage } from "@/pages/experts"
import { LogsPage } from "@/pages/logs"
import { OverviewPage } from "@/pages/overview"
import { PerformancePage } from "@/pages/performance"
import { RequestsPage } from "@/pages/requests"
import { SystemPage } from "@/pages/system"

const PAGES = [
  { id: "overview", title: "Overview", icon: LayoutDashboard, ranged: true },
  { id: "performance", title: "Performance", icon: Zap, ranged: true },
  { id: "experts", title: "Experts", icon: Gauge, ranged: true },
  { id: "system", title: "System", icon: Cpu, ranged: true },
  { id: "requests", title: "Requests", icon: ListOrdered, ranged: false },
  { id: "logs", title: "Logs", icon: ScrollText, ranged: false },
] as const
type PageId = (typeof PAGES)[number]["id"]

function store<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key)
    return v == null ? fallback : (JSON.parse(v) as T)
  } catch {
    return fallback
  }
}
function save(key: string, v: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(v))
  } catch {
    /* private mode: per-viewer conveniences only */
  }
}

function pageFromHash(): PageId {
  const h = window.location.hash.slice(1)
  return (PAGES.find((p) => p.id === h)?.id ?? "overview") as PageId
}

export default function App() {
  const [page, setPage] = useState<PageId>(pageFromHash)
  const [range, setRange] = useState<Range>(() => store("bnk.range", "5m" as Range))
  const [dark, setDark] = useState<boolean>(() => store("bnk.dark", true))
  const [chatOpen, setChatOpen] = useState(false)
  const [messages, setMessages] = useState<Msg[]>([])
  const defaults = useTelemetry((s) => s.overview?.defaults)
  const [settings, setSettings] = useState<ChatSettings>({ temperature: 0.7, top_p: 0.95, max_tokens: 2048, thinking: true })

  useEffect(() => {
    const f = () => setPage(pageFromHash())
    window.addEventListener("hashchange", f)
    return () => window.removeEventListener("hashchange", f)
  }, [])
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    save("bnk.dark", dark)
  }, [dark])
  useEffect(() => save("bnk.range", range), [range])
  useEffect(() => {
    if (defaults) setSettings((s) => ({ ...s, top_p: defaults.top_p }))
  }, [defaults])

  const current = PAGES.find((p) => p.id === page)!
  return (
    <TooltipProvider delayDuration={150}>
      {/* fixed-height shell: only the page content scrolls, so the inset's margins never scroll the window */}
      <SidebarProvider className="h-svh overflow-hidden" style={{ "--sidebar-width": "15rem" } as React.CSSProperties}>
        <AppSidebar page={page} />
        <SidebarInset className="min-h-0 overflow-hidden">
          <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4 lg:px-6">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mx-1 data-[orientation=vertical]:h-4" />
            <h1 className="text-base font-medium">{current.title}</h1>
            <ConnectionBadge />
            <div className="ml-auto flex items-center gap-2">
              {current.ranged && (
                <ToggleGroup type="single" variant="outline" size="sm" value={range} onValueChange={(v) => v && setRange(v as Range)} className="hidden sm:flex">
                  {(Object.keys(RANGES) as Range[]).map((r) => (
                    <ToggleGroupItem key={r} value={r} className="px-2.5">{r}</ToggleGroupItem>
                  ))}
                </ToggleGroup>
              )}
              <Button variant="ghost" size="icon" onClick={() => setDark((d) => !d)} aria-label="Toggle theme">
                {dark ? <Sun /> : <Moon />}
              </Button>
              <Button size="sm" onClick={() => setChatOpen(true)}>
                <MessageSquare />
                <span className="hidden sm:inline">Chat</span>
              </Button>
            </div>
          </header>
          <main className="min-h-0 flex-1 overflow-y-auto p-4 lg:p-6">
            {page === "overview" && <OverviewPage range={range} />}
            {page === "performance" && <PerformancePage range={range} />}
            {page === "experts" && <ExpertsPage range={range} />}
            {page === "system" && <SystemPage range={range} />}
            {page === "requests" && <RequestsPage />}
            {page === "logs" && <LogsPage />}
          </main>
        </SidebarInset>
        <ChatSheet open={chatOpen} onOpenChange={setChatOpen} messages={messages} setMessages={setMessages} settings={settings} setSettings={setSettings} />
      </SidebarProvider>
    </TooltipProvider>
  )
}

function AppSidebar({ page }: { page: PageId }) {
  const model = useTelemetry((s) => s.overview?.model)
  const live = useTelemetry((s) => s.live)
  const uptime = useTelemetry((s) => s.overview?.uptime)
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href="#overview">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <Zap className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">bnk</span>
                  <span className="truncate text-xs text-muted-foreground">{model ?? "inference engine"}</span>
                </div>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {PAGES.map((p) => (
                <SidebarMenuItem key={p.id}>
                  <SidebarMenuButton asChild isActive={page === p.id} tooltip={p.title}>
                    <a href={`#${p.id}`}>
                      <p.icon />
                      <span>{p.title}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="group-data-[collapsible=icon]:hidden">
        <div className="grid gap-1 rounded-lg border p-3 text-xs text-muted-foreground">
          <div className="flex justify-between"><span>Context</span><span className="tabular text-foreground">{fmt.ctx(live?.n_ctx)}</span></div>
          <div className="flex justify-between"><span>Speculation</span><span className="text-foreground">{live?.mtp ? "MTP" : "off"}</span></div>
          <div className="flex justify-between"><span>Uptime</span><span className="tabular text-foreground">{uptime != null ? fmt.duration(uptime) : "—"}</span></div>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}

function ConnectionBadge() {
  const conn = useTelemetry((s) => s.conn)
  const phase = useTelemetry((s) => s.live?.phase)
  if (conn !== "open")
    return (
      <Badge variant="outline" className="gap-1.5">
        <span className="size-1.5 rounded-full bg-warning" />
        {conn === "connecting" ? "Connecting" : "Reconnecting"}
      </Badge>
    )
  return (
    <Badge variant="outline" className="gap-1.5">
      <span className={`size-1.5 rounded-full bg-good ${phase && phase !== "idle" ? "animate-pulse" : ""}`} />
      Live
    </Badge>
  )
}
