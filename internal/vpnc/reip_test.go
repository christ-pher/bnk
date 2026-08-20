package vpnc_test

import (
	"context"
	"encoding/json"
	"net/netip"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/vpnc"
)

// Changing the mesh network on the server must re-address running
// clients: they pick the new prefix up from the netmap, rebuild the
// tunnel, and rejoin without anyone touching the machine.
func TestClientReAddressesWhenServerNetworkChanges(t *testing.T) {
	tc := startControl(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stateA := t.TempDir()
	a := runClient(t, ctx, tc, "alpha", stateA, tc.enrollKey)
	_, firstIP := a.get(t)
	if !netip.MustParsePrefix("100.64.0.0/10").Contains(firstIP) {
		t.Fatalf("initial IP %v is outside the default network", firstIP)
	}

	hc := statusClient(stateA)
	waitRunning(t, hc, true)

	newNet := netip.MustParsePrefix("100.71.0.0/16")
	if err := tc.srv.SetNetwork(newNet); err != nil {
		t.Fatal(err)
	}

	// Wait on the interface, not on status: status echoes the netmap and
	// would go green before the tunnel is actually rebuilt.
	deadline := time.Now().Add(30 * time.Second)
	for {
		if _, ip := a.peek(); newNet.Contains(ip) {
			break
		}
		if time.Now().After(deadline) {
			_, ip := a.peek()
			t.Fatalf("tunnel interface still on %v, want an address in %v", ip, newNet)
		}
		time.Sleep(200 * time.Millisecond)
	}

	// And the daemon must be serving again on the new address.
	deadline = time.Now().Add(30 * time.Second)
	for {
		resp, err := hc.Get("http://bnk/status")
		if err == nil {
			var st vpnc.Status
			jerr := json.NewDecoder(resp.Body).Decode(&st)
			resp.Body.Close()
			if jerr == nil && st.Running && newNet.Contains(st.Self.IP) {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("status never reported running on %v", newNet)
		}
		time.Sleep(200 * time.Millisecond)
	}

	// The new address must also be persisted, so a restart comes back on
	// the new network rather than the old one.
	st, _, err := vpnc.LoadStateForTest(stateA)
	if err != nil {
		t.Fatal(err)
	}
	if !newNet.Contains(st.IP) {
		t.Errorf("persisted IP %v is not in the new network %v", st.IP, newNet)
	}
	if st.Prefix.Bits() != newNet.Bits() {
		t.Errorf("persisted prefix %v does not match the new network %v", st.Prefix, newNet)
	}
}
