# Security model and known weaknesses

Audited 2026-08-21 across four areas: cryptography and authentication,
the remote network surface, local privilege boundaries, and the supply
chain plus policy enforcement. This documents what bnk actually
guarantees, what it does not, and how much each gap deserves to worry
you.

## What bnk assumes

- **The control server is trusted.** It assigns addresses, distributes
  peer keys, and compiles the ACL. Whoever controls it controls the
  mesh. Protect it accordingly.
- **Mesh members are semi-trusted.** WireGuard stops a member from
  spoofing another member's address, but a member sees every other
  node's key, address, and endpoints, and can reach any node the
  receiving node's filter admits.
- **The machine running the client is trusted.** The daemon runs as
  root or SYSTEM. Anyone who is already root there has already won.

## What it genuinely protects against

- **Passive network observers.** Traffic is WireGuard: ChaCha20-Poly1305
  with forward secrecy. The relay carries ciphertext it cannot read.
- **Server impersonation.** The enrollment key carries a SHA-256
  fingerprint of the server's certificate, checked against the leaf on
  every connection. No CA is involved and no domain is needed.
- **Node impersonation.** A session requires proving possession of the
  node's private key against a fresh challenge. Knowing a node's public
  key is not enough.
- **Address spoofing between members.** Each peer is configured with a
  single-address AllowedIPs, so wireguard-go drops a forged source
  before the filter ever sees it.
- **Unauthorized joins.** Enrollment keys are 128-bit, single-use by
  default, expiring, revocable, and compared in constant time.

## What it does not protect against

**The ACL is advisory, not a boundary.** Policy is compiled by the
server and enforced by the *receiving client*. A member running a
modified binary simply ignores it. Treat the ACL as a way to keep honest
nodes in their lane, not as a control that contains a hostile member.
Every node also learns every other node's identity and endpoints
regardless of policy.

**A compromised control server owns the mesh.** It can hand a node any
peer set, any address, and an empty ruleset.

**Client-side compromise is total.** The node's private key is on disk.
Anyone who can read it can be that node from anywhere until it is
removed with `bnk-server node rm`.

## Fixed in this audit

| Severity | Issue |
|---|---|
| **Critical** | The certificate pin accepted a fingerprint match anywhere in the presented chain. Since the server's certificate is public to anyone who connects, an on-path attacker could terminate TLS with their own key, append the real certificate, and pass the check — full MITM of enrollment and every coordination session. Only the leaf is compared now, in constant time. `internal/pin/pin_chain_test.go` performs the actual attack. |
| High | The internet-facing port had no header timeout, idle timeout, body cap, or read deadline on the first frame after upgrade. Unauthenticated connections could be held open indefinitely or made to allocate arbitrarily. |
| High | Hostname, OS, and advertised endpoints were rebroadcast to every peer without bounds. One node could exceed the netmap frame limit and take the mesh down, or aim every peer's probes at a third party. |
| Medium | A plaintext `http://` server URL was accepted, silently disabling the trust bootstrap. Now refused. |

## Known weaknesses, in the order they deserve attention

### 1. Windows: the node's private key is readable by any local user

`%ProgramData%\bnk\client.json` holds the node's private key. Go's
`Chmod` and `MkdirAll` modes are no-ops on Windows, and `C:\ProgramData`
grants `BUILTIN\Users` read by inheritance. Nothing sets a DACL.

Anyone who can run code as *any* local user — including malware running
as you, without elevation — can copy that key and impersonate the node
from anywhere, until it is removed server-side. Linux is unaffected
(`/var/lib/bnk/client.json` is 0600 root).

**Worry level: real on a shared or malware-exposed Windows machine.**
The fix is to set an explicit DACL on the state directory at install.

### 2. Enrollment does not enforce unique names or disco keys

Node names are chosen by the enrolling node and the ACL is keyed by
name, last one wins. A member holding a valid enrollment key can enroll
as `admin-laptop`, inherit that name's permissions as a *source*, and
lock the real one out. `node rm` matches by name and would delete the
wrong node. Duplicate disco keys similarly let a member hijack another
peer's path attribution.

**Worry level: near zero while every node is yours; high the moment
someone else's machine joins.** Fix before sharing the mesh: reject
duplicate names and disco keys at enrollment, and remove nodes by ID.

### 3. A member can make every other member flood a third party

Advertised endpoints are now capped at 32, which bounds this, but each
peer still probes whatever addresses a member advertises, with no
filtering of private or off-mesh ranges. A hostile member can still
direct modest reflected traffic at a chosen host.

**Worry level: low for you, but it is collateral risk to strangers.**
Filter advertised endpoints against bogon and private ranges.

### 4. Unauthenticated disco floods can stall a client

Inbound disco packets are decrypted before checking whether the sender
is a known peer, and handled inline on the receive path. Roughly 15k
packets per second to a client's WireGuard port can starve it enough to
stop passing traffic. The port must be known, which a mesh member always
knows.

**Worry level: low unless someone is deliberately targeting you.** Look
up the sender before doing the key exchange.

### 5. Windows local privilege details

`bnk leave` reads state directly rather than going through the gated
API, so on Windows — where the state file is readable (weakness 1) — any
local user can deregister the machine. The tray's operator account can
also call `/join`, which re-points the daemon at a different control
server. Both are narrower than weakness 1, which grants the key itself.

**Worry level: follows from weakness 1.** Fixing the DACL removes most
of it; routing `leave` through the control pipe removes the rest.

### 6. Releases are trusted because GitHub is trusted

`SHA256SUMS` is generated by the same job that builds the artifacts and
published to the same release, unsigned. It detects a corrupted
download, not a malicious one. The install one-liners fetch scripts from
`main` unpinned, and the Windows tray downloads its MSI with no checksum
at all. Anyone who can push to the repository, or who steals the release
token, gets root on every machine at the next update.

**Worry level: proportional to your GitHub account's security.** Enable
two-factor authentication and treat that account as the mesh's root of
trust. Signing (Sigstore for binaries, Authenticode for Windows) is the
real fix; see the code-signing notes in DEPLOY.md.

### 7. Smaller items

- Removing a node that a policy group still references makes the policy
  fail to compile, which fails *closed* — the whole mesh goes deny-all
  with nothing in the logs explaining why. Re-validate policy on node
  removal.
- The stateful filter refreshes flow expiry on inbound packets, so one
  outbound packet to a hostile node lets it hold the return channel open
  indefinitely. `icmp` in a policy permits every ICMP type, not just
  echo.
- The filter does not check the destination address, so on a host with
  IP forwarding enabled (Docker turns this on globally) a member can
  reach that host's LAN.
- No rate limiting anywhere: STUN, relay, and the local diagnostics API
  are all unbounded.

## If you take three actions

1. Turn on two-factor authentication for the GitHub account. It is the
   root of trust for every binary on every machine.
2. Set a restrictive ACL on `%ProgramData%\bnk` on Windows machines.
3. Before adding anyone else's machine, fix enrollment uniqueness — and
   remember the ACL will still not contain a hostile member.
