//go:build linux

package vpnc

// DefaultSocket is where the daemon serves the local API. It lives under
// /run (not the 0700 state dir) so status works without root.
const DefaultSocket = "/run/bnk/bnk.sock"

// DefaultStateDir holds the node's identity and enrollment result.
const DefaultStateDir = "/var/lib/bnk"
