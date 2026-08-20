package main

import (
	"net/netip"
	"reflect"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/vpnc"
)

func TestStatusRowsMergesSelfSortedByID(t *testing.T) {
	now := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	st := vpnc.Status{
		Self: vpnc.SelfStatus{ID: 2, Name: "bravo", IP: netip.MustParseAddr("100.64.0.2")},
		Peers: []vpnc.PeerStatus{
			{ID: 3, Name: "charlie", IP: netip.MustParseAddr("100.64.0.3"), Online: true, Path: "direct", LastHandshake: now.Add(-5 * time.Second)},
			{ID: 1, Name: "alpha", IP: netip.MustParseAddr("100.64.0.1"), Online: false, Path: "relay"},
		},
	}
	got := statusRows(st, now)
	want := [][]string{
		{"alpha", "100.64.0.1", "false", "relay", "never"},
		{"bravo*", "100.64.0.2", "true", "-", "-"},
		{"charlie", "100.64.0.3", "true", "direct", "5s ago"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("statusRows =\n%v\nwant\n%v", got, want)
	}
}
