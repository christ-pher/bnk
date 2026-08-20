# Deploy: VPS control plane + Linux clients

Total time: ~10 minutes (3 build, 3 VPS, 2 per client, 2 verify).

What you end up with: every machine gets a stable `100.64.x.x` address and
can reach the others over encrypted WireGuard, directly when possible,
relayed through the VPS when not.

```
[client A] ──┐
             ├── VPS :8443 (enroll, coordination, relay, STUN)
[client B] ──┘         then A ⇄ B directly when a path exists
```

Everything lands in predefined places — no paths to choose:

| What | Where |
|---|---|
| Binaries | `/usr/local/bin/{vpnd,vpn}` |
| Server state (cert, registry, admin socket) | `/var/lib/vpnd` |
| Client config | `/etc/vpnmesh/vpn.env` |
| Client identity/state | `/var/lib/vpn` |
| Client local API socket (no sudo needed) | `/run/vpnmesh/vpn.sock` |
| systemd units | `vpnd.service`, `vpn.service` |

## Before you start

- [ ] A VPS with a public IP (any small Linux box works)
- [ ] Root on the VPS and on each client machine
- [ ] Go 1.22+ on ONE machine to build (binaries are static; build once, copy anywhere)

Build both binaries now (~2 min):

```
git clone <your-repo> vpnmesh && cd vpnmesh
CGO_ENABLED=0 go build -o vpnd ./cmd/vpnd
CGO_ENABLED=0 go build -o vpn ./cmd/vpn
```

## Part 1 — VPS (3 min)

1. Copy the binary and installer, then run it:
   ```
   scp vpnd install-server.sh root@YOUR_VPS:
   ssh root@YOUR_VPS './install-server.sh'
   ```
   It installs the binary and the systemd unit, starts the service, and
   prints the cert fingerprint.
2. Open ONE port, both protocols (this is the only firewall change anywhere):
   ```
   ufw allow 8443/tcp && ufw allow 8443/udp     # or your provider's firewall panel
   ```
   TCP = enrollment/coordination/relay. UDP = STUN. Same number on purpose.
3. Mint one enrollment key PER CLIENT (keys are single-use and die in 24h —
   that's the join security):
   ```
   ssh root@YOUR_VPS 'vpnd key new'
   ```
   Copy the whole `vpnkey:...` line somewhere. Run it again for the second
   client. Done with the VPS.

## Part 2 — each client (2 min per machine)

```
scp vpn install-client.sh root@CLIENT:
ssh root@CLIENT './install-client.sh --server https://YOUR_VPS_IP:8443 --key vpnkey:PASTE_HERE'
```

The installer starts the service, waits for the tunnel, prints
`vpn status`, and blanks the spent key from `/etc/vpnmesh/vpn.env`
automatically. Repeat with the OTHER key on the second client.

## Part 3 — verify (2 min, run on client A, no sudo needed)

Work down this list; each line proves one layer:

```
vpn status               # all nodes listed (yours marked *), ONLINE true
ping 100.64.0.2          # traffic flows (relay or direct)
vpn ping NODENAME        # direct path + RTT, if punchable
vpn netcheck             # relay counters, path state, probe log
```

How to read it:
- `PATH direct` = machines talk directly; the VPS only coordinates.
- `PATH relay` + working ping = fine too; traffic relays through the VPS.
  Two cloud VPSes usually go direct. A machine behind a hostile NAT may not.
- `vpn ping` timing out while `ping` works = relay-only peer. Expected
  behind port-randomizing NATs; the error text says so.

Real throughput test: `iperf3 -s` on B, `iperf3 -c 100.64.0.2` on A.

## Everyday commands

```
vpn status | vpn ping NAME | vpn netcheck   # diagnostics, no sudo
vpn down                                    # disconnect (daemon stays; survives reboot)
vpn up                                      # reconnect
vpnd up / vpnd down                         # start/stop the control server (on the VPS)
```

## Locking it down (optional, 5 min)

No policy = every node reaches every node. To restrict, on the VPS:

1. Write `policy.json` (start from `policy.example.json` in the repo —
   groups of node names, then from/to/allow rules like `tcp/22`).
2. Apply and dry-run-check it:
   ```
   vpnd acl set policy.json
   vpnd acl check nodeA nodeB tcp/22
   ```
Rules take effect on all clients within seconds. Replies to connections a
node initiates are always allowed; everything else not listed is dropped.

## Key management (VPS, as root)

```
vpnd key ls                        # what exists, what's spent
vpnd key new --reusable --ttl 1h   # multi-node key, if you must
vpnd key revoke PREFIX             # kill a key by its ls prefix
vpnd node ls                       # who's enrolled + online
```

## When something doesn't work

| Symptom | Cause → fix |
|---|---|
| `enroll: 403` | Key spent, expired, or mistyped → mint a new one on the VPS |
| client log: `certificate does not match pinned fingerprint` | vpnd state dir was rebuilt (new cert) → re-enroll client with a fresh key |
| `vpn status` says down but you didn't run `vpn down` | Service stopped → `systemctl start vpn`; the down/up state persists on purpose |
| peer `ONLINE false` | Its vpn service is down, or its network path to the VPS is → check `journalctl -u vpn` there |
| ping works, then dies for ~15s, then works | Paths renegotiating; check `vpn netcheck`'s `disco_events` and send me the output |
| nothing works | `journalctl -u vpnd` on the VPS; is 8443/tcp+udp actually open? `curl -k https://VPS:8443/enroll` should answer (405/400, not timeout) |

Next after it works: add your laptop the same way — that's the machine
where the NAT traversal actually earns its keep.
