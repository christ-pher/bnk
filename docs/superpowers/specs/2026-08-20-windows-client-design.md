# Windows client for bnk — design

Status: approved 2026-08-20. Scope: Windows **client** only. `bnk-server`
stays Linux-only. macOS remains deferred.

## Goal

`bnk` runs on Windows as a service, joins the mesh, and behaves like the
Linux client: `bnk status` works unelevated, `bnk up` / `bnk down` need
Administrator, and a single install command joins a new machine.

## What already works

Verified before design, not assumed:

- `GOOS=windows go build ./...` passes on the current tree.
- wireguard-go's Wintun backend exports `CreateTUN(name, mtu)` with the
  same signature as Linux, so `realTUN()` in `cmd/bnk` is unchanged.
- The dataplane (magicsock, disco, filter, relay, coord client, ACL
  filter, netmap) is platform-independent. UDP batching is already gated
  to Linux and falls back to one datagram per syscall elsewhere.
- `osName()` reports `windows`, so `bnk-server node ls` shows it.

## Components

### 1. `internal/router/router_windows.go`

Applies the tunnel's IP, route, and MTU via
`golang.zx2c4.com/wireguard/windows/tunnel/winipcfg` (new dependency,
v1.0.1):

- `luid.SetIPAddresses([]netip.Prefix{prefix})` — assigning the mesh
  prefix creates the on-link route, matching how the Linux
  implementation relies on `ip addr add 100.64.x.y/10`.
- `luid.IPInterface(AddressFamilyIPv4)` → set `NLMTU` → `Set()`.

**Interface change:** `router.Up` currently takes an interface *name*.
Windows needs the adapter's LUID, which comes from the `tun.Device`
object. New signature:

```go
func Up(dev tun.Device, ifName string, prefix netip.Prefix, mtu int) error
```

Linux ignores `dev` and keeps shelling out to `ip`; Windows type-asserts
`dev.(*tun.NativeTun)` and reads `LUID()`. `router_other.go` keeps
returning the unsupported-platform error for everything else.

### 2. Local API transport split

**Problem.** The daemon exposes a local HTTP API for the CLI. On Linux it
is one unix socket at `/run/bnk/bnk.sock`, mode 0666 so `bnk status`
needs no sudo, with `POST /up` and `POST /down` gated by `SO_PEERCRED`
(root or the daemon's uid). Windows has no `SO_PEERCRED`.

**Solution.** Windows expresses the same policy with named-pipe ACLs, so
the routes are split by privilege level and mounted per platform.

Handler construction splits into two functions in `internal/vpnc`:

- `diagnosticsHandler(c *controller)` — `GET /status`, `/ping`, `/netcheck`
- `controlHandler(c *controller)` — `POST /up`, `/down`

Listeners are platform files:

- **Linux** (`localapi_listen_linux.go`): one unix socket serving both
  handler sets, with the existing `SO_PEERCRED` gate on the control
  routes. Behavior is byte-for-byte what ships today.
- **Windows** (`localapi_listen_windows.go`): two named pipes via
  `github.com/Microsoft/go-winio` (new dependency, v0.6.2):
  - `\\.\pipe\bnk` — SDDL grants Everyone connect access; serves
    diagnostics only.
  - `\\.\pipe\bnk-ctl` — SDDL grants Administrators and SYSTEM only;
    serves control only. Windows refuses the open for anyone else, so no
    impersonation or in-process identity check is needed.

The CLI dials the diagnostics pipe for read-only verbs and the control
pipe for `up`/`down`, via a per-platform dialer.

**Rejected:** a single admin-only pipe (loses unelevated `bnk status`,
the property this project deliberately added on Linux); AF_UNIX on
Windows (permission model depends on NTFS ACLs that still require
Windows API calls, and is a far less-trodden path than pipes).

### 3. Windows service integration (`cmd/bnk`)

- `bnk run` calls `svc.IsWindowsService()`. Under the SCM it runs through
  `svc.Run` with a handler that reports Start/Running and cancels the
  daemon context on Stop/Shutdown. Run interactively, it behaves exactly
  as today.
- New subcommands, Windows-only: `bnk service install --server URL
  [--key bnkkey:...]` and `bnk service uninstall`, registering
  `bnk.exe run --server URL [--key ...] --state-dir <ProgramData>\bnk`
  as an auto-start service named `bnk` running as LocalSystem.
- The enrollment key is single-use. Mirroring the Linux installer, which
  blanks `BNK_KEY` in the env file after first successful start, the
  PowerShell installer re-runs `bnk service install --server URL`
  (without `--key`) once the tunnel is up, rewriting the service
  arguments so a spent key is never resubmitted.

### 4. Platform paths

Per-OS constants replace hardcoded Linux paths:

| | Linux | Windows |
|---|---|---|
| Binary | `/usr/local/bin/bnk` | `C:\Program Files\bnk\bnk.exe` |
| State | `/var/lib/bnk` | `%ProgramData%\bnk` |
| Local API | `/run/bnk/bnk.sock` | `\\.\pipe\bnk` (+ `-ctl`) |

`vpnc.DefaultSocket` and a new `vpnc.DefaultStateDir` move into
`paths_linux.go` / `paths_windows.go`; `cmd/bnk` uses them for flag
defaults instead of literals. Windows resolves `%ProgramData%` from the
environment rather than assuming `C:`.

### 5. Release and installer

**CI** adds `windows/amd64` and `windows/arm64`. The release job
downloads `wintun-0.14.1.zip` from wintun.net, verifies a pinned SHA256,
and publishes `bnk-windows-<arch>.zip` containing `bnk.exe`,
`wintun.dll`, and Wintun's `LICENSE.txt`.

Wintun's prebuilt-binary license permits redistribution "insofar as the
Software is distributed alongside other software that uses the Software
only via the Permitted API" (clause 3d) and forbids extracting from it
(3a). Shipping the DLL as a sibling file is the permitted case, and it
matches what Tailscale (`fullyQualifiedWintunPath()` loads it from the
executable's directory) and NetBird do. **Do not embed it in the exe.**

A separate CI job runs `GOOS=windows go build ./...` on every push so
cross-compilation cannot silently regress.

**`install-client.ps1`** mirrors `install-client.sh`: downloads the zip
for the machine's architecture, extracts to `C:\Program Files\bnk`,
registers and starts the service, waits for the tunnel, scrubs the key,
and supports `-Uninstall`. Invocation:

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.ps1))) -Server https://VPS:8443 -Key bnkkey:...
```

`bnk-server key new` prints both the Linux and Windows join commands,
labeled, since the server cannot know the target platform.

## Out of scope for v1

- **MSI installer.** Planned as the immediate follow-up once the client
  is validated on real hardware; it changes no Go code. Debugging client
  bugs through an MSI layer is materially harder than through a script,
  so the script proves the client first.
- **`bnk update` on Windows.** The release asset is a zip and swapping a
  running `.exe` needs rename-then-restart handling. The PowerShell
  installer is idempotent, so re-running it is the update path; `bnk
  update` on Windows prints that instruction rather than half-working.
- **`bnk-server` on Windows** and **macOS** support.

## Testing

Runnable in CI and locally on Linux:

- Handler split: diagnostics reachable, control gated (existing local API
  tests must keep passing unchanged).
- Per-OS path constants and service-argument construction (pure
  functions, table tests).
- `GOOS=windows go build ./...` and `go vet` as a CI job.

Requires the user's Windows machine (cannot be tested here):

1. Install via the PowerShell one-liner; service registered and running.
2. `bnk status` **unelevated** lists nodes with self marked `*`.
3. `ping` a Linux peer's `100.64.x.y` address; traffic flows.
4. `bnk ping <peer>` reports a direct path or explains relay-only.
5. `bnk down` / `bnk up` from an **elevated** prompt succeed; from an
   unelevated prompt they are refused.
6. Reboot: service auto-starts and rejoins.
7. `-Uninstall` removes service, binaries, and state.

## Known risks

- **Throughput.** Wintun's TUN layer has no batching upstream (`TODO:
  implement batching with wintun`), so Windows will trail Linux
  throughput. Expected, not a defect.
- **SmartScreen.** `bnk.exe` is unsigned; Windows may warn on first run.
  Code signing needs a certificate and is out of scope. Documented.
- **Untestable-here surface.** Wintun adapter creation, SCM lifecycle,
  and pipe ACLs are validated only by the checklist above.
