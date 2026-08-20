//go:build linux

package main

import (
	"context"
	"net"
)

const daemonDownHint = "bnk daemon not running — start it with: systemctl start bnk"

func dialLocal(ctx context.Context, sock string) (net.Conn, error) {
	var d net.Dialer
	return d.DialContext(ctx, "unix", sock)
}

// controlEndpoint is the same unix socket on Linux: control verbs are
// gated there by the caller's uid rather than by a separate endpoint.
func controlEndpoint(sock string) string { return sock }
