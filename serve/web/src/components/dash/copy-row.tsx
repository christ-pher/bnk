import { Check, Copy } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

// The clipboard API only exists on https or localhost; over plain http (the dashboard opened by IP) fall back to
// a hidden textarea and execCommand.
async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    /* fall through */
  }
  const ta = document.createElement("textarea")
  ta.value = text
  ta.setAttribute("readonly", "")
  ta.style.position = "fixed"
  ta.style.opacity = "0"
  document.body.appendChild(ta)
  ta.select()
  let ok = false
  try {
    ok = document.execCommand("copy")
  } catch {
    ok = false
  }
  document.body.removeChild(ta)
  return ok
}

// A label, a value (truncated, full text on hover) and a copy button.
export function CopyRow({ label, value }: { label: string; value: string | undefined }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="grid min-w-0 gap-0.5">
      <span>{label}</span>
      <div className="flex min-w-0 items-center gap-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-foreground">{value ?? "—"}</span>
          </TooltipTrigger>
          {value && <TooltipContent side="right">{value}</TooltipContent>}
        </Tooltip>
        <Button
          variant="ghost"
          size="icon"
          className="size-6 shrink-0"
          disabled={!value}
          aria-label={`Copy ${label}`}
          onClick={async () => {
            if (value && (await copyText(value))) {
              setCopied(true)
              window.setTimeout(() => setCopied(false), 1500)
            }
          }}
        >
          {copied ? <Check className="size-3.5 text-good" /> : <Copy className="size-3.5" />}
        </Button>
      </div>
    </div>
  )
}
