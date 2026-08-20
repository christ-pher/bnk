# Windows tray app for bnk — design

Status: approved 2026-08-20. Windows only. Ships **before** the MSI, so
the MSI packages the tray in one pass instead of being reworked.

## Goal

Toggle the VPN and see the mesh from the system tray, with no elevated
prompt.

## The problem this has to solve

The control pipe (`\\.\pipe\bnk-ctl`) is ACL'd to Administrators and
SYSTEM. Under UAC a non-elevated process carries the Administrators SID
as *deny-only*, so a tray app running as the logged-in user cannot open
that pipe at all — which is precisely why `bnk up` / `bnk down` need an
elevated prompt today.

A tray toggle is therefore an **authorization** change, not a GUI
problem. The chosen model is an operator account, as Tailscale does with
`tailscale set --operator=`.

## Components

### 1. Operator SID (`internal/vpnc`, `cmd/bnk`)

- `vpnc.Config.OperatorSID`, set by `bnk run --operator <SID>`.
- The Windows listener builds the control pipe's descriptor as SYSTEM +
  Administrators + the operator SID. With no operator configured the
  descriptor is exactly what ships today, so Linux and unconfigured
  Windows installs are unaffected.
- `controlSDDL(sid)` is a pure function with table tests, and rejects
  anything that is not a well-formed `S-1-...` SID rather than splicing
  caller-supplied text into an ACL.
- `bnk service install --operator <SID>` records it in the service's
  arguments, alongside `--server`.

**This widens control from "any administrator" to "any administrator
plus one named account, unelevated."** That is inherent to the request;
the operator model keeps it to one account rather than all local users.

### 2. Shared local-API client (`internal/localclient`)

`cmd/bnk` and the tray both need to call the local API, so the dialing
and request helpers move into one package rather than being duplicated.
It holds the per-OS dialer (unix socket / named pipe), `Status`, `Up`,
and `Down`.

### 3. `cmd/bnk-tray` (new binary, Windows only)

A separate binary because `-H=windowsgui`, which suppresses the console
window, applies to the whole binary and would break `bnk.exe`'s CLI
output.

- Polls `Status` on the diagnostics pipe every 3s.
- Menu:

  ```
  ● Connected — 100.64.0.4
  ─────────────────────────
  Disconnect
  Peers                ▸   alpha    100.64.0.1  direct
  Copy my IP               beta     100.64.0.2  relay
  ─────────────────────────  racknerd 100.64.0.3  offline
  Quit
  ```

- The action item flips between Connect and Disconnect; clicking a peer
  copies its address.
- `systray` cannot remove menu items, so the Peers submenu holds a fixed
  pool of entries whose titles are rewritten each poll, with unused ones
  hidden. Meshes larger than the pool get a trailing "…and N more".
- A refused toggle (tray running as a non-operator) surfaces the reason
  in the status line instead of failing silently.
- Quit exits only the tray; the VPN keeps running, because the tray is a
  remote control, not the daemon.
- Two embedded `.ico` files distinguish connected from disconnected.

Dependency: `fyne.io/systray` v1.12.2 — verified to cross-compile to
windows/amd64 and arm64 with CGO disabled.

### 4. Installer and packaging

- `bnk-tray.exe` joins `bnk-windows-<arch>.zip`.
- `install-client.ps1` gains `-Operator` (defaulting to the installing
  user's SID), passes it to `bnk service install`, installs the tray,
  adds an HKCU Run entry so it starts at login, and launches it.
- `-Uninstall` removes the Run entry and stops any running tray.

## Testing

Runnable here: `controlSDDL` construction and SID validation, the
status-to-menu-label logic, the peer-row formatter, and `GOOS=windows`
builds of every binary in CI.

Requires the Windows machine:

1. Tray appears at login with the right state.
2. **Unelevated** Connect/Disconnect works — the critical check, since
   it proves the operator ACL.
3. Peers submenu lists the mesh and matches `bnk status`.
4. `bnk down` from an unelevated *shell* is still refused (only the
   operator's tray path is permitted, not the world).
5. Uninstall removes the tray and its Run entry.

## Out of scope

Notifications/balloons, connection statistics, editing settings from the
tray, and any macOS or Linux tray.
