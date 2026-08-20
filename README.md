# vpnmesh

A deliberately simple, open-source mesh VPN built on WireGuard, in Go.
One CLI control server, one client binary, no web UI, no accounts.

**Status: functional on Linux, NAT traversal validated.** Working today:
control plane (enroll, IP assignment, netmap push), encrypted relay
through the server, port/protocol ACLs enforced by a userspace packet
filter, and NAT traversal — clients behind NATs start on the relay, then
hole-punch (STUN + authenticated disco probes) and upgrade to a direct
path, falling back to relay when the punch is impossible (e.g. two
port-randomizing NATs). Both outcomes are exercised against real
netfilter NATs in `test/natlab/`. macOS/Windows clients compile but lack
interface configuration for now.

Diagnostics: `vpn status` (per-peer path: direct/relay), `vpn ping <peer>`
(disco-level RTT), `vpn netcheck` (local + STUN-observed endpoints).

## How it works

- `vpnd` runs on a publicly reachable server: enrolls nodes, assigns IPs
  from `100.64.0.0/10`, and pushes peer maps over one TLS connection.
- `vpn` runs on each machine: brings up a userspace WireGuard interface
  (wireguard-go) and keeps it configured from the server's pushes.
- Trust is a pinned certificate fingerprint embedded in the enrollment
  key — no CA, no domain required.

**Deploying for real?** Follow [DEPLOY.md](DEPLOY.md) — VPS control plane
plus clients via the install scripts, ~10 minutes.

## Quickstart

Server (any Linux box with a public IP):

```
sudo ./install-server.sh     # installs vpnd + systemd unit, prints next steps
sudo vpnd key new            # prints vpnkey:<secret>:<fingerprint>, one per client
```

Each client (Linux):

```
sudo ./install-client.sh --server https://YOUR_SERVER:8443 --key vpnkey:...
```

The enrollment key is single-use; the node's identity persists in
`/var/lib/vpn`. Day to day: `vpn status` (no sudo needed) shows every
node's path (direct or relay) and handshake age, `vpn down` / `vpn up`
disconnect and reconnect, `vpnd node ls` on the server shows the mesh,
and `vpnd down` / `vpnd up` stop and start the control server.

Access control (see `policy.example.json`):

```
vpnd acl set policy.json
vpnd acl check laptop nas tcp/22   # dry-run: allowed or denied?
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
