package vpnc_test

// Full-production-codepath reproduction of the natlab symmetric failure:
// two vpnc.Run clients over TLS whose advertised endpoints are all
// unreachable. Hole punching cannot succeed; the relay must carry the
// tunnel anyway.

import (
	"context"
	"log"
	"net/netip"
	"path/filepath"
	"testing"

	"github.com/christ-pher/bnk/internal/vpnc"
)

func runDeadEndClient(t *testing.T, ctx context.Context, tc *testControl, name string, deadEp netip.AddrPort) *tnetHolder {
	t.Helper()
	h := &tnetHolder{}
	stateDir := t.TempDir()
	go func() {
		err := vpnc.Run(ctx, vpnc.Config{
			ServerURL:         tc.url,
			EnrollKey:         tc.enrollKey,
			StateDir:          stateDir,
			SocketPath:        filepath.Join(stateDir, "bnk.sock"),
			Hostname:          name,
			CreateTUN:         h.factory,
			Logf:              log.Printf,
			EndpointsOverride: []netip.AddrPort{deadEp},
		})
		if err != nil && ctx.Err() == nil {
			t.Errorf("%s Run: %v", name, err)
		}
	}()
	return h
}

func TestRunRelayCarriesTunnelWhenEndpointsAreDead(t *testing.T) {
	tc := startControl(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	a := runDeadEndClient(t, ctx, tc, "alpha", netip.MustParseAddrPort("192.0.2.10:41641"))
	b := runDeadEndClient(t, ctx, tc, "beta", netip.MustParseAddrPort("192.0.2.20:41641"))
	echoOverTunnel(t, a, b)
}
