//go:build windows

package localclient

import (
	"context"
	"net"

	"github.com/Microsoft/go-winio"

	"github.com/christ-pher/bnk/internal/vpnc"
)

const DaemonDownHint = "bnk service not running — start it with: sc start bnk"

func dial(ctx context.Context, pipe string) (net.Conn, error) {
	return winio.DialPipeContext(ctx, pipe)
}

// ControlEndpoint returns the restricted pipe. Opening it requires
// Administrator or the configured operator account.
func ControlEndpoint(pipe string) string { return vpnc.ControlPipe(pipe) }
