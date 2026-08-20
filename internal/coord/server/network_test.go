package server_test

import (
	"net/netip"
	"testing"

	"github.com/christ-pher/bnk/internal/netmap"
)

func TestSetNetworkReassignsNodesAndReportsPrefix(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "alpha", ident32(t, 1).pub)
	e.enroll(t, "beta", ident32(t, 2).pub)

	if got := e.srv.Network(); got.String() != "100.64.0.0/10" {
		t.Fatalf("default network = %v, want 100.64.0.0/10", got)
	}

	if err := e.srv.SetNetwork(netip.MustParsePrefix("100.67.0.0/16")); err != nil {
		t.Fatal(err)
	}
	if got := e.srv.Network(); got.String() != "100.67.0.0/16" {
		t.Errorf("network = %v, want 100.67.0.0/16", got)
	}

	// Host numbers are preserved, so .1 and .2 carry over.
	newPrefix := netip.MustParsePrefix("100.67.0.0/16")
	for _, n := range adminNodes(t, e) {
		if !newPrefix.Contains(n.IP) {
			t.Errorf("node %s kept %v, outside the new network", n.Name, n.IP)
		}
	}
}

// A node that re-enrolls (or a new one) must be given an address from the
// current network, not the original default.
func TestEnrollAfterSetNetworkUsesNewPrefix(t *testing.T) {
	e := startServer(t)
	if err := e.srv.SetNetwork(netip.MustParsePrefix("100.70.0.0/16")); err != nil {
		t.Fatal(err)
	}
	resp := e.enroll(t, "gamma", ident32(t, 3).pub)
	want := netip.MustParsePrefix("100.70.0.0/16")
	if !want.Contains(resp.IP) {
		t.Errorf("enrolled IP %v is outside %v", resp.IP, want)
	}
	if resp.Prefix.String() != want.String() {
		t.Errorf("enroll response prefix = %v, want %v", resp.Prefix, want)
	}
}

func TestSetNetworkRejectsUnusablePrefixes(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "alpha", ident32(t, 1).pub)
	e.enroll(t, "beta", ident32(t, 2).pub)
	e.enroll(t, "gamma", ident32(t, 3).pub)

	// /30 leaves 2 usable addresses for 3 nodes.
	if err := e.srv.SetNetwork(netip.MustParsePrefix("10.9.0.0/30")); err == nil {
		t.Error("expected an error for a prefix too small to hold every node")
	}
	// IPv6 is out of scope for the mesh.
	if err := e.srv.SetNetwork(netip.MustParsePrefix("fd00::/64")); err == nil {
		t.Error("expected an error for an IPv6 prefix")
	}
	// A /31 has no assignable addresses: it must be refused even though
	// an empty mesh would technically "fit" in it.
	empty := startServer(t)
	if err := empty.srv.SetNetwork(netip.MustParsePrefix("10.9.0.0/31")); err == nil {
		t.Error("expected an error for a prefix with no assignable addresses")
	}
	// The rejected attempts must not have changed anything.
	if got := e.srv.Network(); got.String() != "100.64.0.0/10" {
		t.Errorf("network = %v after failed changes, want the original", got)
	}
}

// The netmap must carry the mesh prefix, not a bare /32: it is what tells
// a client to re-address its interface after the network changes.
func TestNetmapSelfIPCarriesMeshPrefix(t *testing.T) {
	e := startServer(t)
	id := ident32(t, 1)
	e.enroll(t, "alpha", id.pub)
	if err := e.srv.SetNetwork(netip.MustParsePrefix("100.67.0.0/16")); err != nil {
		t.Fatal(err)
	}

	_, nms := dialSession(t, e, id.priv)
	nm := waitNetmap(t, nms, func(nm netmap.Netmap) bool {
		return netip.MustParsePrefix("100.67.0.0/16").Contains(nm.SelfIP.Addr())
	})
	if nm.SelfIP.Bits() != 16 {
		t.Errorf("netmap SelfIP = %v, want a /16 to match the mesh prefix", nm.SelfIP)
	}
	if !netip.MustParsePrefix("100.67.0.0/16").Contains(nm.SelfIP.Addr()) {
		t.Errorf("netmap SelfIP %v is outside the new network", nm.SelfIP)
	}
}
