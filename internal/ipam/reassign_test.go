package ipam_test

import (
	"net/netip"
	"testing"

	"github.com/christ-pher/bnk/internal/ipam"
)

func addrs(ss ...string) []netip.Addr {
	out := make([]netip.Addr, len(ss))
	for i, s := range ss {
		out[i] = netip.MustParseAddr(s)
	}
	return out
}

// Keeping each node's host number makes the change predictable: a node
// at .3 stays at .3 in the new network.
func TestReassignPreservesHostOffset(t *testing.T) {
	got, err := ipam.Reassign(
		netip.MustParsePrefix("100.64.0.0/10"),
		netip.MustParsePrefix("100.67.0.0/16"),
		addrs("100.64.0.1", "100.64.0.2", "100.64.0.7"),
	)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"100.64.0.1": "100.67.0.1",
		"100.64.0.2": "100.67.0.2",
		"100.64.0.7": "100.67.0.7",
	}
	for from, to := range want {
		if g := got[netip.MustParseAddr(from)]; g.String() != to {
			t.Errorf("%s -> %s, want %s", from, g, to)
		}
	}
}

// An offset past the end of a smaller network has to be packed back in.
func TestReassignPacksOffsetsThatNoLongerFit(t *testing.T) {
	got, err := ipam.Reassign(
		netip.MustParsePrefix("100.64.0.0/10"),
		netip.MustParsePrefix("100.67.0.0/29"), // 6 usable
		addrs("100.64.0.1", "100.64.5.9"),      // second offset is way out of range
	)
	if err != nil {
		t.Fatal(err)
	}
	if g := got[netip.MustParseAddr("100.64.0.1")]; g.String() != "100.67.0.1" {
		t.Errorf("in-range address moved to %s, want 100.67.0.1", g)
	}
	moved := got[netip.MustParseAddr("100.64.5.9")]
	if !netip.MustParsePrefix("100.67.0.0/29").Contains(moved) {
		t.Errorf("out-of-range address mapped to %s, outside the new prefix", moved)
	}
	if moved == netip.MustParseAddr("100.67.0.1") {
		t.Error("packed address collided with a preserved one")
	}
}

func TestReassignRejectsTooSmallPrefix(t *testing.T) {
	_, err := ipam.Reassign(
		netip.MustParsePrefix("100.64.0.0/10"),
		netip.MustParsePrefix("100.67.0.0/30"), // 2 usable
		addrs("100.64.0.1", "100.64.0.2", "100.64.0.3"),
	)
	if err == nil {
		t.Fatal("expected an error when the new prefix cannot hold every node")
	}
}

func TestReassignIsIdenticalWhenPrefixUnchanged(t *testing.T) {
	p := netip.MustParsePrefix("100.64.0.0/10")
	got, err := ipam.Reassign(p, p, addrs("100.64.0.1", "100.64.9.9"))
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range addrs("100.64.0.1", "100.64.9.9") {
		if got[a] != a {
			t.Errorf("%s moved to %s, want unchanged", a, got[a])
		}
	}
}
