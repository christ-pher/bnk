import { ArrowUp, Brain, ChevronRight, Eraser, Settings2, Square } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Slider } from "@/components/ui/slider"
import { Spinner } from "@/components/ui/spinner"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { fmt } from "@/lib/format"
import { useTelemetry } from "@/lib/telemetry"

export interface Msg {
  role: "user" | "assistant"
  content: string
  reasoning?: string
  pending?: boolean
  error?: string
  timings?: { tps?: number; prefill_tps?: number; tokens_per_round?: number; expert_miss_rate?: number; prompt_tokens?: number }
  tokens?: number
}

export interface ChatSettings {
  temperature: number
  top_p: number
  max_tokens: number
  thinking: boolean
}

export function ChatSheet({
  open,
  onOpenChange,
  messages,
  setMessages,
  settings,
  setSettings,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  messages: Msg[]
  setMessages: React.Dispatch<React.SetStateAction<Msg[]>>
  settings: ChatSettings
  setSettings: React.Dispatch<React.SetStateAction<ChatSettings>>
}) {
  const model = useTelemetry((s) => s.overview?.model)
  const [input, setInput] = useState("")
  const [showSettings, setShowSettings] = useState(false)
  const abort = useRef<AbortController | null>(null)
  const end = useRef<HTMLDivElement>(null)
  const busy = messages.some((m) => m.pending)

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" })
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    setInput("")
    const history = [...messages.filter((m) => !m.error), { role: "user" as const, content: text }]
    setMessages([...history, { role: "assistant", content: "", reasoning: "", pending: true }])
    const ctl = new AbortController()
    abort.current = ctl
    const update = (f: (m: Msg) => Msg) =>
      setMessages((ms) => {
        const c = ms.slice()
        c[c.length - 1] = f(c[c.length - 1])
        return c
      })
    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Bnk-Client": "dashboard" },
        body: JSON.stringify({
          messages: history.map(({ role, content }) => ({ role, content })),
          stream: true,
          stream_options: { include_usage: true },
          temperature: settings.temperature,
          top_p: settings.top_p,
          max_tokens: settings.max_tokens,
          chat_template_kwargs: { enable_thinking: settings.thinking },
        }),
        signal: ctl.signal,
      })
      if (!res.ok || !res.body) throw new Error((await res.text()) || `HTTP ${res.status}`)
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ""
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        let i
        while ((i = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, i)
          buf = buf.slice(i + 2)
          for (const line of chunk.split("\n")) {
            if (!line.startsWith("data: ")) continue
            const data = line.slice(6)
            if (data === "[DONE]") continue
            const j = JSON.parse(data)
            const d = j.choices?.[0]?.delta ?? {}
            if (d.content || d.reasoning_content)
              update((m) => ({ ...m, content: m.content + (d.content ?? ""), reasoning: (m.reasoning ?? "") + (d.reasoning_content ?? "") }))
            if (j.timings) update((m) => ({ ...m, timings: j.timings }))
            if (j.usage) update((m) => ({ ...m, tokens: j.usage.completion_tokens }))
          }
        }
      }
      update((m) => ({ ...m, pending: false }))
    } catch (e) {
      const aborted = (e as Error).name === "AbortError"
      update((m) => ({ ...m, pending: false, error: aborted ? undefined : String((e as Error).message ?? e) }))
    } finally {
      abort.current = null
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-xl">
        <SheetHeader className="border-b">
          <SheetTitle>Test chat</SheetTitle>
          <SheetDescription className="truncate">{model ?? "bnk"} · requests appear in the dashboard as “dashboard”</SheetDescription>
          <div className="flex gap-2 pt-1">
            <Button variant={showSettings ? "secondary" : "outline"} size="sm" onClick={() => setShowSettings((v) => !v)}>
              <Settings2 />
              Settings
            </Button>
            <Button variant="outline" size="sm" disabled={busy || !messages.length} onClick={() => setMessages([])}>
              <Eraser />
              Clear
            </Button>
          </div>
        </SheetHeader>

        {showSettings && (
          <FieldGroup className="gap-4 border-b p-4">
            <Field>
              <div className="flex justify-between">
                <FieldLabel>Temperature</FieldLabel>
                <span className="text-sm tabular text-muted-foreground">{settings.temperature.toFixed(2)}</span>
              </div>
              <Slider min={0} max={2} step={0.05} value={[settings.temperature]} onValueChange={([v]) => setSettings((s) => ({ ...s, temperature: v }))} />
            </Field>
            <Field>
              <div className="flex justify-between">
                <FieldLabel>Top-p</FieldLabel>
                <span className="text-sm tabular text-muted-foreground">{settings.top_p.toFixed(2)}</span>
              </div>
              <Slider min={0.05} max={1} step={0.01} value={[settings.top_p]} onValueChange={([v]) => setSettings((s) => ({ ...s, top_p: v }))} />
            </Field>
            <div className="flex items-end gap-6">
              <Field className="w-36">
                <FieldLabel htmlFor="max-tokens">Max tokens</FieldLabel>
                <Input id="max-tokens" type="number" min={1} value={settings.max_tokens}
                  onChange={(e) => setSettings((s) => ({ ...s, max_tokens: Math.max(1, Number(e.target.value) || 1) }))} />
              </Field>
              <div className="flex items-center gap-2 pb-2">
                <Switch id="thinking" checked={settings.thinking} onCheckedChange={(v) => setSettings((s) => ({ ...s, thinking: v }))} />
                <Label htmlFor="thinking">Thinking</Label>
              </div>
            </div>
          </FieldGroup>
        )}

        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-1 text-center text-sm text-muted-foreground">
              <span className="font-medium text-foreground">Try the engine</span>
              <span>Speed, acceptance and expert misses show under each reply.</span>
            </div>
          ) : (
            <div className="flex flex-col gap-5">
              {messages.map((m, i) => (m.role === "user" ? <UserMsg key={i} m={m} /> : <AssistantMsg key={i} m={m} />))}
            </div>
          )}
          <div ref={end} />
        </div>

        <form
          className="border-t p-3"
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
        >
          <div className="relative">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
              placeholder="Message the model…"
              className="max-h-48 min-h-20 resize-none pr-12"
            />
            {busy ? (
              <Button type="button" size="icon" variant="secondary" className="absolute right-2 bottom-2 size-8" onClick={() => abort.current?.abort()} aria-label="Stop">
                <Square />
              </Button>
            ) : (
              <Button type="submit" size="icon" className="absolute right-2 bottom-2 size-8" disabled={!input.trim()} aria-label="Send">
                <ArrowUp />
              </Button>
            )}
          </div>
        </form>
      </SheetContent>
    </Sheet>
  )
}

function UserMsg({ m }: { m: Msg }) {
  return (
    <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-md bg-secondary px-3.5 py-2 text-sm whitespace-pre-wrap">{m.content}</div>
  )
}

function AssistantMsg({ m }: { m: Msg }) {
  const thinking = m.pending && !m.content
  return (
    <div className="grid gap-2 text-sm">
      {m.reasoning ? (
        <Collapsible defaultOpen={false}>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
            <ChevronRight className="size-3.5 transition-transform group-data-[state=open]:rotate-90" />
            <Brain className="size-3.5" />
            {thinking ? "Thinking…" : "Reasoning"}
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="mt-2 border-l-2 pl-3 text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">{m.reasoning}</div>
          </CollapsibleContent>
        </Collapsible>
      ) : null}
      {m.pending && !m.content && !m.reasoning && (
        <div className="flex items-center gap-2 text-muted-foreground"><Spinner className="size-3.5" />Reading the prompt…</div>
      )}
      {m.content && (
        <div className="prose-chat leading-relaxed">
          <Markdown remarkPlugins={[remarkGfm]}>{m.content}</Markdown>
        </div>
      )}
      {m.error && <div className="text-critical">{m.error}</div>}
      {!m.pending && m.timings && (
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="outline" className="tabular">{fmt.n(m.timings.tps, 1)} tok/s</Badge>
          {m.tokens != null && <Badge variant="outline" className="tabular">{fmt.n(m.tokens)} tokens</Badge>}
          <Badge variant="outline" className="tabular">prompt {fmt.n(m.timings.prefill_tps)} tok/s</Badge>
          {m.timings.tokens_per_round != null && <Badge variant="outline" className="tabular">{fmt.n(m.timings.tokens_per_round, 2)} tok/round</Badge>}
          {m.timings.expert_miss_rate != null && <Badge variant="outline" className="tabular">{fmt.pct(m.timings.expert_miss_rate, 1)} CPU misses</Badge>}
        </div>
      )}
    </div>
  )
}
