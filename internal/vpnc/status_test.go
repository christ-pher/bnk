package vpnc_test

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"vpnmesh/internal/vpnc"
)

func statusClient(stateDir string) *http.Client {
	sock := filepath.Join(stateDir, "vpn.sock")
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", sock)
		},
	}}
}

func TestStatusReportsPeersAndPath(t *testing.T) {
	tc := startControl(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stateA := t.TempDir()
	a := runClient(t, ctx, tc, "alpha", stateA, tc.enrollKey)
	b := runClient(t, ctx, tc, "beta", t.TempDir(), tc.enrollKey)
	echoOverTunnel(t, a, b)

	hc := statusClient(stateA)
	deadline := time.Now().Add(10 * time.Second)
	for {
		resp, err := hc.Get("http://vpn/status")
		if err == nil {
			defer resp.Body.Close()
			var st vpnc.Status
			if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
				t.Fatal(err)
			}
			if !st.Self.IP.IsValid() {
				t.Error("status has no self IP")
			}
			// The direct path is proven asynchronously by disco, so keep
			// polling until the peer shows up as direct.
			if len(st.Peers) == 1 && st.Peers[0].Name == "beta" && st.Peers[0].Path == "direct" {
				p := st.Peers[0]
				if !p.IP.IsValid() {
					t.Error("peer has no IP")
				}
				if p.LastHandshake.IsZero() {
					t.Error("peer has no handshake time despite live tunnel")
				}
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("status never showed peer beta (last err: %v)", err)
		}
		time.Sleep(200 * time.Millisecond)
	}
}
