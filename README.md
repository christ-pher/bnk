# bnk

A deliberately simple, open-source mesh VPN built on WireGuard, in Go.
One CLI control server, one client binary, no web UI, no accounts.

**Status: functional on Linux, NAT traversal validated.** Working today:
control plane (enroll, IP assignment, netmap push), encrypted relay
through the server, port/protocol ACLs enforced by a userspace packet
filter, and NAT traversal — clients behind NATs start on the relay, then
hole-punch (STUN + authenticated disco probes) and upgrade to a direct
path, falling back to relay when the punch is impossible (e.g. two
port-randomizing NATs). Both outcomes are exercised against real
netfilter NATs in `test/natlab/`. Linux and Windows clients are
supported; macOS is deferred.

Diagnostics: `bnk status` (per-peer path: direct/relay), `bnk ping <peer>`
(disco-level RTT), `bnk netcheck` (local + STUN-observed endpoints).

On Windows a tray icon toggles the tunnel without an elevated prompt and
lists the mesh under Peers.

## How it works

- `bnk-server` runs on a publicly reachable server: enrolls nodes, assigns IPs
  from `100.64.0.0/10`, and pushes peer maps over one TLS connection.
- `bnk` runs on each machine: brings up a userspace WireGuard interface
  (wireguard-go) and keeps it configured from the server's pushes.
- Trust is a pinned certificate fingerprint embedded in the enrollment
  key — no CA, no domain required.

**Deploying for real?** Follow [DEPLOY.md](DEPLOY.md) — VPS control plane
plus clients via the install scripts, ~10 minutes.

## Quickstart

Server (any Linux box with a public IP):

```
curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-server.sh | sudo sh
bnk-server key new           # one key per client; prints a paste-ready install command
```

Each client — paste what `key new` printed for that platform:

```
# Linux (root)
curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.sh | sudo sh -s -- --server https://YOUR_SERVER:8443 --key bnkkey:...
```

```powershell
# Windows (elevated PowerShell)
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.ps1))) -Server https://YOUR_SERVER:8443 -Key bnkkey:...
```

Binaries come from [GitHub Releases](https://github.com/christ-pher/bnk/releases),
built by CI on every version tag: linux amd64/arm64, and windows
amd64/arm64 zipped with Wintun's driver DLL alongside. The Linux
installer also accepts a locally built binary placed next to it.

The enrollment key is single-use; the node's identity persists in
`/var/lib/bnk`. Day to day: `bnk status` (no sudo needed) shows every
node's path (direct or relay) and handshake age, `bnk down` / `bnk up`
disconnect and reconnect, `bnk-server node ls` on the server shows the mesh,
and `bnk-server down` / `bnk-server up` stop and start the control server.

Access control (see `policy.example.json`):

```
bnk-server acl set policy.json
bnk-server acl check laptop nas tcp/22   # dry-run: allowed or denied?
```

No policy means allow-all; an explicit policy is default-deny — only the
listed flows (plus replies to connections a node initiates) pass.

## Development

```
go test -race ./...          # unit + in-process integration (no root)
sudo test/e2e/smoke.sh       # real binaries + TUNs in network namespaces
sudo test/natlab/punch.sh    # hole punch through two masquerading NATs
sudo test/natlab/punch.sh symmetric  # hostile NATs: relay fallback
```

The integration tests run entire meshes in-process using netstack TUNs —
control server, enrollment, and WireGuard tunnels — with no privileges.
