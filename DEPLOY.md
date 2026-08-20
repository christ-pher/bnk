# Deploy: VPS control plane + two Linux clients

Total time: ~25 minutes (10 VPS, 5 per client, 5 verify).

What you end up with: every machine gets a stable `100.64.x.x` address and
can reach the others over encrypted WireGuard, directly when possible,
relayed through the VPS when not.

```
[client A] ──┐
             ├── VPS :8443 (enroll, coordination, relay, STUN)
[client B] ──┘         then A ⇄ B directly when a path exists
```

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

## Part 1 — VPS (10 min)

1. Copy the server binary up:
   ```
   scp vpnd root@YOUR_VPS:/usr/local/bin/vpnd
   ```
2. Open ONE port, both protocols (this is the only firewall change anywhere):
   ```
   ufw allow 8443/tcp && ufw allow 8443/udp     # or your provider's firewall panel
   ```
   TCP = enrollment/coordination/relay. UDP = STUN. Same number on purpose.
3. Install the service:
   ```
   scp contrib/systemd/vpnd.service root@YOUR_VPS:/etc/systemd/system/
   ssh root@YOUR_VPS 'systemctl daemon-reload && systemctl enable --now vpnd'
   ```
4. Confirm it's up — you want a line containing `cert fingerprint`:
   ```
   ssh root@YOUR_VPS 'journalctl -u vpnd -n 3'
   ```
5. Mint one enrollment key PER CLIENT (keys are single-use and die in 24h —
   that's the join security):
   ```
   ssh root@YOUR_VPS 'vpnd key new --state-dir /var/lib/vpnd'
   ```
   Copy the whole `vpnkey:...` line somewhere. Run it again for the second
   client. Done with the VPS.

## Part 2 — each client (5 min per machine)

1. Copy the client binary:
   ```
   scp vpn root@CLIENT:/usr/local/bin/vpn
   ```
2. Create the config file (paste one of your minted keys):
   ```
   mkdir -p /etc/vpnmesh
   cat > /etc/vpnmesh/vpn.env <<EOF
   VPN_SERVER=https://YOUR_VPS_IP:8443
   VPN_KEY=vpnkey:PASTE_THE_WHOLE_THING_HERE
   EOF
   chmod 600 /etc/vpnmesh/vpn.env
   ```
3. Install and start:
   ```
   cp contrib/systemd/vpn.service /etc/systemd/system/
   systemctl daemon-reload && systemctl enable --now vpn
   ```
4. Confirm — you want `up: node N, ip 100.64.0.N`:
   ```
   journalctl -u vpn -n 3
   ```
5. After the first successful start, blank the key line in
   `/etc/vpnmesh/vpn.env` (`VPN_KEY=`). The machine is enrolled; its
   identity lives in `/var/lib/vpn/` now and the key is spent anyway.

Repeat for the second client with the OTHER key.

## Part 3 — verify (5 min, run on client A)

Work down this list; each line proves one layer:

```
vpn status --state-dir /var/lib/vpn        # peer listed, ONLINE true
ping 100.64.0.2                            # traffic flows (relay or direct)
vpn ping --state-dir /var/lib/vpn NODENAME # direct path + RTT, if punchable
vpn netcheck --state-dir /var/lib/vpn      # relay counters, path state, probe log
```

How to read it:
- `PATH direct` = machines talk directly; the VPS only coordinates.
- `PATH relay` + working ping = fine too; traffic relays through the VPS.
  Two cloud VPSes usually go direct. A machine behind a hostile NAT may not.
- `vpn ping` timing out while `ping` works = relay-only peer. Expected
  behind port-randomizing NATs; the error text says so.

Real throughput test: `iperf3 -s` on B, `iperf3 -c 100.64.0.2` on A.
Expect noticeably less than line rate (userspace WireGuard) — tens to
hundreds of Mbit/s depending on CPU.

## Locking it down (optional, 5 min)

No policy = every node reaches every node. To restrict, on the VPS:

1. Write `policy.json` (start from `policy.example.json` in the repo —
   groups of node names, then from/to/allow rules like `tcp/22`).
2. Apply and dry-run-check it:
   ```
   vpnd acl set --state-dir /var/lib/vpnd policy.json
   vpnd acl check --state-dir /var/lib/vpnd nodeA nodeB tcp/22
   ```
Rules take effect on all clients within seconds. Replies to connections a
node initiates are always allowed; everything else not listed is dropped.

## Key management (VPS)

```
vpnd key ls --state-dir /var/lib/vpnd              # what exists, what's spent
vpnd key new --reusable --ttl 1h --state-dir ...   # multi-node key, if you must
vpnd key revoke --state-dir /var/lib/vpnd PREFIX   # kill a key by its ls prefix
vpnd node ls --state-dir /var/lib/vpnd             # who's enrolled + online
```

## When something doesn't work

| Symptom | Cause → fix |
|---|---|
| `enroll: 403` | Key spent, expired, or mistyped → mint a new one on the VPS |
| client log: `certificate does not match pinned fingerprint` | vpnd state dir was rebuilt (new cert) → re-enroll client with a fresh key |
| peer `ONLINE false` | Its vpn service is down, or its network path to the VPS is → check `journalctl -u vpn` there |
| ping works, then dies for ~15s, then works | Paths renegotiating; check `vpn netcheck`'s `disco_events` and send me the output |
| nothing works | `journalctl -u vpnd` on the VPS; is 8443/tcp+udp actually open? `curl -k https://VPS:8443/enroll` should answer (405/400, not timeout) |

Next after it works: add your laptop the same way — that's the machine
where the NAT traversal actually earns its keep.
