//go:build linux

package localclient

import (
	"context"
	"net"
)

// DaemonDownHint tells the user how to start the daemon on this OS.
const DaemonDownHint = "bnk daemon not running — start it with: systemctl start bnk"

func dial(ctx context.Context, sock string) (net.Conn, error) {
	var d net.Dialer
	return d.DialContext(ctx, "unix", sock)
}

// ControlEndpoint is the same unix socket on Linux: control verbs are
// gated there by the caller's uid rather than by a separate endpoint.
func ControlEndpoint(sock string) string { return sock }
