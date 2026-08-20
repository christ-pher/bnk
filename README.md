# vpnmesh

A deliberately simple, open-source mesh VPN built on WireGuard, in Go.
One CLI control server, one client binary, no web UI, no accounts.

**Status: early. Phase 1 (control plane) works on Linux.** Peers connect
directly when reachable; NAT traversal (hole punching) and relay fallback
are planned — see `docs/` plan. macOS/Windows clients compile but lack
interface configuration for now.

## How it works

- `vpnd` runs on a publicly reachable server: enrolls nodes, assigns IPs
  from `100.64.0.0/10`, and pushes peer maps over one TLS connection.
- `vpn` runs on each machine: brings up a userspace WireGuard interface
  (wireguard-go) and keeps it configured from the server's pushes.
- Trust is a pinned certificate fingerprint embedded in the enrollment
  key — no CA, no domain required.

## Quickstart

Server (any Linux box with a public IP):

```
vpnd serve --state-dir /var/lib/vpnd --listen :8443
vpnd key new --state-dir /var/lib/vpnd   # prints vpnkey:<secret>:<fingerprint>
```

Each client (Linux, as root):

```
vpn up --server https://YOUR_SERVER:8443 --key vpnkey:...
```

The enrollment key is only needed the first time; after that `vpn up
--server ...` resumes from saved state. `vpnd node ls` shows the mesh.

## Development

```
go test -race ./...          # unit + in-process integration (no root)
sudo test/e2e/smoke.sh       # real binaries + TUNs in network namespaces
```

The integration tests run entire meshes in-process using netstack TUNs —
control server, enrollment, and WireGuard tunnels — with no privileges.
