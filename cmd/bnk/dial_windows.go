//go:build windows

package main

import (
	"context"
	"net"

	"github.com/Microsoft/go-winio"

	"github.com/christ-pher/bnk/internal/vpnc"
)

const daemonDownHint = "bnk service not running — start it with: sc start bnk"

func dialLocal(ctx context.Context, pipe string) (net.Conn, error) {
	return winio.DialPipeContext(ctx, pipe)
}

// controlEndpoint returns the admin-only pipe. Opening it fails for a
// non-elevated caller, which is how Windows enforces what SO_PEERCRED
// enforces on Linux.
func controlEndpoint(pipe string) string { return vpnc.ControlPipe(pipe) }
