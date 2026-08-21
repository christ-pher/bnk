# Wishlist

Things worth doing, not yet done. Ordered roughly by value.

## Sign-in prompt: a bare key has no server in it

The tray's sign-in box invites either the whole join command or just the
`bnkkey:...` value, but those are not equivalent. The whole command
carries the server URL; a bare key does not. Pasting only the key works
when the daemon already knows its server — re-joining after being
removed — and fails on a fresh install, where there is nothing to
connect to.

Today that surfaces as an error after the fact ("no server known: paste
the whole join command, which includes it"), which is accurate but
arrives too late to be helpful.

Options, roughly in order of preference:

- Have the prompt adapt: the daemon knows whether it has a server, so
  `Status` could report it and the tray could ask for exactly what is
  missing — the key alone when the server is known, both when it is not.
- Ask for the server in a second box, only when the pasted text lacks
  one and the daemon has none.
- A real two-field dialog. Best result, most work: it means building a
  Win32 dialog rather than borrowing PowerShell's InputBox.

## `bnk-server key new` output is cluttered

The framed block with two labelled platform sections is more furniture
than the content needs. It should stay copy-pasteable — single-line
commands, since a wrapped line survives a triple-click and a backslash
continuation does not — but it can lose most of the framing and prose.

Worth considering: print only what was asked for, with a flag to select
a platform, rather than both every time.

## Smaller items

- `bnk update` on Windows is a stub pointing at the installer. Now that
  the tray can fetch and launch the MSI, the CLI could do the same.
- The security audit's remaining findings are listed in SECURITY.md
  under "Known weaknesses" — notably filter flow-expiry refreshed by
  inbound packets, ICMP allowing every type rather than echo, and no
  rate limiting on STUN, the relay, or the local API.
- Reboot persistence on Windows has still never been tested end to end.
