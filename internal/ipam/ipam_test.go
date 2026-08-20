package ipam

import (
	"net/netip"
	"testing"
)

func addrs(ss ...string) map[netip.Addr]bool {
	m := make(map[netip.Addr]bool)
	for _, s := range ss {
		m[netip.MustParseAddr(s)] = true
	}
	return m
}

func TestNextAllocatesFirstHostAddress(t *testing.T) {
	got, err := Next(netip.MustParsePrefix("100.64.0.0/10"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := netip.MustParseAddr("100.64.0.1"); got != want {
		t.Errorf("Next = %v, want %v", got, want)
	}
}

func TestNextSkipsUsedAddresses(t *testing.T) {
	used := addrs("100.64.0.1", "100.64.0.2", "100.64.0.4")
	got, err := Next(netip.MustParsePrefix("100.64.0.0/10"), used)
	if err != nil {
		t.Fatal(err)
	}
	if want := netip.MustParseAddr("100.64.0.3"); got != want {
		t.Errorf("Next = %v, want %v", got, want)
	}
}

func TestNextErrorsWhenPrefixExhausted(t *testing.T) {
	// A /30 has two usable hosts (.1 and .2); both taken.
	used := addrs("192.168.0.1", "192.168.0.2")
	if got, err := Next(netip.MustParsePrefix("192.168.0.0/30"), used); err == nil {
		t.Errorf("Next = %v, want exhaustion error", got)
	}
}
