package vpnc

// Regression for the tunnel-loop bug: a node must never advertise its own
// tunnel address as an endpoint candidate — probing it routes into the
// tunnel itself, "proves" a looping path, and strangles the tunnel.

import (
	"net/netip"
	"testing"
)

func TestFilterEndpointsExcludesMeshPrefix(t *testing.T) {
	mesh := netip.MustParsePrefix("100.64.0.0/10")
	in := []netip.AddrPort{
		netip.MustParseAddrPort("192.168.1.5:41641"),
		netip.MustParseAddrPort("100.64.0.1:41641"), // own tunnel addr: must go
		netip.MustParseAddrPort("100.99.7.7:41641"), // also inside 100.64/10
		netip.MustParseAddrPort("127.0.0.1:41641"),
	}
	got := filterEndpoints(in, mesh)
	if len(got) != 2 {
		t.Fatalf("filtered = %v, want the two non-mesh endpoints", got)
	}
	for _, ep := range got {
		if mesh.Contains(ep.Addr()) {
			t.Errorf("mesh address %v survived the filter", ep)
		}
	}
}

func TestFilterEndpointsNoPrefixKeepsAll(t *testing.T) {
	in := []netip.AddrPort{netip.MustParseAddrPort("100.64.0.1:1")}
	if got := filterEndpoints(in, netip.Prefix{}); len(got) != 1 {
		t.Errorf("invalid prefix must filter nothing, got %v", got)
	}
}
