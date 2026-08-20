# Deploy: VPS control plane + Linux clients

Total time: ~5 minutes (2 VPS, 1 per client, 2 verify). No build step —
binaries come from GitHub Releases.

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
| Binaries | `/usr/local/bin/{bnk-server,bnk}` |
| Server state (cert, registry, admin socket) | `/var/lib/bnk-server` |
| Client config | `/etc/bnk/bnk.env` |
| Client identity/state | `/var/lib/bnk` |
| Client local API socket (no sudo needed) | `/run/bnk/bnk.sock` |
| systemd units | `bnk-server.service`, `bnk.service` |

## Before you start

- [ ] A VPS with a public IP (any small Linux box works)
- [ ] Root on the VPS and on each client machine

No build step: the installers pull prebuilt binaries (linux amd64/arm64)
from GitHub Releases. Working from a source checkout instead? Put a
locally built binary next to the script and it uses that.

## Part 1 — VPS (2 min)

1. Install the control server (one line, as root on the VPS):
   ```
   curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-server.sh | sudo sh
   ```
   It downloads the binary, installs the systemd unit, starts the
   service, and prints the cert fingerprint.
2. Open ONE port, both protocols (this is the only firewall change anywhere):
   ```
   ufw allow 8443/tcp && ufw allow 8443/udp     # or your provider's firewall panel
   ```
   TCP = enrollment/coordination/relay. UDP = STUN. Same number on purpose.
3. Mint one enrollment key PER CLIENT (keys are single-use and die in 24h —
   that's the join security):
   ```
   bnk-server key new
   ```
   Below the key it prints a ready-to-paste install command with the
   server URL and key already filled in. Run it again for each client.
   Done with the VPS.

   (If the VPS is behind NAT or the detected IP looks wrong, set
   `--public-url https://REAL_IP:8443` on `bnk-server serve` in the unit.)

## Part 2 — each client (1 min per machine)

Paste the command `key new` printed — it prints one for each platform.

**Linux**, as root:

```
curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.sh | sudo sh -s -- --server https://YOUR_VPS_IP:8443 --key bnkkey:PASTE_HERE
```

**Windows**, from an elevated PowerShell:

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.ps1))) -Server https://YOUR_VPS_IP:8443 -Key bnkkey:PASTE_HERE
```

Either installer downloads the binary, registers and starts the service,
waits for the tunnel, prints `bnk status`, and scrubs the spent key
automatically. Repeat per client with its own key.

Windows notes: the client installs to `C:\Program Files\bnk` (add it to
your PATH to run `bnk` from anywhere) with state in `%ProgramData%\bnk`.
`bnk status` works unelevated; `bnk up` / `bnk down` need an elevated
prompt. `bnk.exe` is unsigned, so SmartScreen may warn on first run.
Re-running the installer is how you update a Windows client — `bnk
update` is Linux-only for now. Throughput trails Linux because Wintun
has no batched I/O upstream.

## Part 3 — verify (2 min, run on client A, no sudo needed)

Work down this list; each line proves one layer:

```
bnk status               # all nodes listed (yours marked *), ONLINE true
ping 100.64.0.2          # traffic flows (relay or direct)
bnk ping NODENAME        # direct path + RTT, if punchable
bnk netcheck             # relay counters, path state, probe log
```

How to read it:
- `PATH direct` = machines talk directly; the VPS only coordinates.
- `PATH relay` + working ping = fine too; traffic relays through the VPS.
  Two cloud VPSes usually go direct. A machine behind a hostile NAT may not.
- `bnk ping` timing out while `ping` works = relay-only peer. Expected
  behind port-randomizing NATs; the error text says so.

Real throughput test: `iperf3 -s` on B, `iperf3 -c 100.64.0.2` on A.

## Everyday commands

```
bnk status | bnk ping NAME | bnk netcheck   # diagnostics, no sudo
bnk down                                    # disconnect (daemon stays; survives reboot)
bnk up                                      # reconnect
sudo bnk update                             # upgrade to the latest release in place (state kept)
bnk-server status                           # is the control server up? + node summary (VPS)
bnk-server up / bnk-server down             # start/stop the control server (VPS)
sudo bnk-server update                      # upgrade the control server in place (state kept)
```

`update` downloads the latest release for this machine's architecture,
verifies it against the release checksums, swaps the binary, and restarts
the service. Nothing else is touched — identities, keys, and config stay.
`bnk version` / `bnk-server version` show what's currently installed.

Uninstall completely (same script, `-u`; deletes state and identity too):

```
curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.sh | sudo sh -s -- -u   # client
curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-server.sh | sudo sh -s -- -u  # server
```

## Locking it down (optional, 5 min)

No policy = every node reaches every node. To restrict, on the VPS:

1. Write `policy.json` (start from `policy.example.json` in the repo —
   groups of node names, then from/to/allow rules like `tcp/22`).
2. Apply and dry-run-check it:
   ```
   bnk-server acl set policy.json
   bnk-server acl check nodeA nodeB tcp/22
   ```
Rules take effect on all clients within seconds. Replies to connections a
node initiates are always allowed; everything else not listed is dropped.

## Removing a node

Uninstalling a client deregisters it automatically — the installer runs
`bnk leave` before it removes anything, so the node disappears from
`bnk-server node ls` on its own.

That needs the server to be reachable at uninstall time. When it isn't —
or when the machine died, was reimaged, or you just want to evict it —
remove it from the VPS:

```
bnk-server node ls            # find the name
bnk-server node rm laptop     # forget it; its address returns to the pool
```

A machine still running bnk after removal cannot rejoin on its own: the
server no longer has its key, so its log says it was removed from the
mesh and it needs a fresh key from `bnk-server key new`.

## Changing the mesh network (VPS, as root)

The mesh uses `100.64.0.0/10` by default. To move it somewhere else:

```
bnk-server net get                      # what it is now
bnk-server net set 100.67.0.0/16        # confirms first; --yes to skip
```

Every node is re-addressed, keeping its host number where it fits
(`100.64.0.3` becomes `100.67.0.3`). Connected clients pick the change up
from the next netmap, rebuild their tunnel, and rejoin on their own;
nodes that are offline do it when they reconnect. **Traffic drops for a
few seconds** while tunnels restart, and anything pinned to an old
address (scripts, `/etc/hosts`, firewall rules) needs updating.

A network must be IPv4, `/30` or larger, and big enough for every
enrolled node — otherwise the change is refused and nothing changes.

## Key management (VPS, as root)

```
bnk-server key ls                        # what exists, what's spent
bnk-server key new --reusable --ttl 1h   # multi-node key, if you must
bnk-server key revoke PREFIX             # kill a key by its ls prefix
bnk-server node ls                       # who's enrolled + online
```

## When something doesn't work

| Symptom | Cause → fix |
|---|---|
| `enroll: 403` | Key spent, expired, or mistyped → mint a new one on the VPS |
| client log: `certificate does not match pinned fingerprint` | bnk-server state dir was rebuilt (new cert) → re-enroll client with a fresh key |
| `bnk status` says down but you didn't run `bnk down` | Service stopped → `systemctl start bnk`; the down/up state persists on purpose |
| peer `ONLINE false` | Its bnk service is down, or its network path to the VPS is → check `journalctl -u bnk` there |
| ping works, then dies for ~15s, then works | Paths renegotiating; check `bnk netcheck`'s `disco_events` and send me the output |
| nothing works | `journalctl -u bnk-server` on the VPS; is 8443/tcp+udp actually open? `curl -k https://VPS:8443/enroll` should answer (405/400, not timeout) |

Next after it works: add your laptop the same way — that's the machine
where the NAT traversal actually earns its keep.
