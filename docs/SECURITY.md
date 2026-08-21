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

## Fixed since the audit

| Date | Severity | Issue |
|---|---|---|
| 2026-08-20 | High | `%ProgramData%\bnk` inherited a `BUILTIN\Users`-readable DACL, so any local user could copy the node's private key and impersonate the node from anywhere. The state directory now gets an explicit SYSTEM+Administrators DACL at install (`internal/vpnc/harden_windows.go`), asserted in CI. |
| 2026-08-20 | High | Enrollment accepted duplicate names and disco keys, letting a key-holder enroll as an existing node and inherit its ACL identity or capture its path attribution. Both are rejected now (`internal/coord/server/unique_test.go`); re-enrollment is allowed only for the same node key. |
| 2026-08-20 | Medium | Inbound disco packets were decrypted before the sender was looked up, so unauthenticated junk cost a key exchange each. The lookup now happens first; `Open` still proves the claim. |
| 2026-08-21 | High | A live enrollment key was committed to this public repository in a pasted PowerShell transcript (`windows-ps-output.txt`, from commit `47f41e4`). The file is removed, but the key is in public git history and must be treated as burned: revoke it (`bnk-server key revoke <prefix>`) and audit `bnk-server node ls` for machines you do not recognise. Keys minted before expiry existed never expire, which makes rotation the only off switch. |
| 2026-08-21 | Medium | The tray downloaded its MSI with no checksum and handed it to msiexec. It is now fetched through `selfupdate.FetchVerified`, which refuses anything that does not match the release's `SHA256SUMS` — the same gate the Linux binary path always had. |
| 2026-08-21 | Medium | The `ci` and `windows` workflows ran with the default `GITHUB_TOKEN` grants. Both now declare `permissions: contents: read`, so a compromised step cannot push, tag, or write releases with the workflow's token. |

## GitHub code scanning triage (2026-08-21)

Code scanning was enabled and produced seven alerts. Six were the
workflow-permissions findings fixed above. The seventh —
`go/disabled-certificate-check` against `internal/pin/pin.go`, severity
high — is a false positive: `InsecureSkipVerify` there is paired with a
`VerifyPeerCertificate` that *replaces* chain and hostname verification
with a constant-time SHA-256 pin of the leaf certificate, which is the
entire trust design (no CA, no domains; the fingerprint travels in the
enrollment key). `TestClientTLSConfigRejectsWrongFingerprint` and
`TestPinRejectsRealCertAppendedToAttackerChain` prove the property
CodeQL cannot see. Dismiss the alert with that justification rather than
"fixing" it — routing this through the WebPKI would weaken the design,
not strengthen it.

## Known weaknesses, in the order they deserve attention

### 1. A member can make every other member flood a third party

Advertised endpoints are capped at 32, which bounds this, but each peer
still probes whatever addresses a member advertises, with no filtering
of private or off-mesh ranges. A hostile member can still direct modest
reflected traffic at a chosen host.

**Worry level: low for you, but it is collateral risk to strangers.**
Filter advertised endpoints against bogon and private ranges.

### 2. Windows local privilege details

`bnk leave` reads state directly rather than going through the gated
API. The state directory DACL now stops ordinary users, but the tray's
operator account can still call `/join`, which re-points the daemon at a
different control server.

**Worry level: low.** Routing `leave` through the control pipe and
narrowing what the operator may pass to `/join` would close the rest.

### 3. Releases are trusted because GitHub is trusted

`SHA256SUMS` is generated by the same job that builds the artifacts and
published to the same release, unsigned. Every installer and updater now
checks it, but it still detects a corrupted download, not a malicious
release: anyone who can push to the repository, or who steals the
release token, gets root on every machine at the next update. The
install one-liners also fetch scripts from `main` unpinned.

**Worry level: proportional to your GitHub account's security.** Enable
two-factor authentication and treat that account as the mesh's root of
trust. Signing (Sigstore for binaries, Authenticode for Windows) is the
real fix; see the code-signing notes in DEPLOY.md.

### 4. Smaller items

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

1. Revoke the leaked enrollment key and audit the node list. The key is
   in public git history; assume it has been read.
2. Turn on two-factor authentication for the GitHub account, plus secret
   scanning with push protection so the next pasted transcript is caught
   at push time. It is the root of trust for every binary on every
   machine.
3. Before adding anyone else's machine, remember the ACL will still not
   contain a hostile member.
