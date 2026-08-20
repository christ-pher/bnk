package vpnc

import (
	"context"
	"errors"
	"net/netip"
	"sync"
	"testing"
	"time"
)

// NAT mappings drift (and get poisoned); the client must re-query STUN
// periodically and re-advertise only when the observed address changes.
func TestStunRefreshLoopAdvertisesOnChange(t *testing.T) {
	a1 := netip.MustParseAddrPort("203.0.113.1:1000")
	a2 := netip.MustParseAddrPort("203.0.113.1:2000")
	answers := []struct {
		ap  netip.AddrPort
		err error
	}{
		{a1, nil}, {a1, nil}, {netip.AddrPort{}, errors.New("blip")}, {a2, nil},
	}

	var mu sync.Mutex
	var got []netip.AddrPort
	i := 0
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go stunRefreshLoop(ctx, 5*time.Millisecond,
		func(context.Context) (netip.AddrPort, error) {
			mu.Lock()
			defer mu.Unlock()
			a := answers[i]
			if i < len(answers)-1 {
				i++
			}
			return a.ap, a.err
		},
		func(ap netip.AddrPort) {
			mu.Lock()
			defer mu.Unlock()
			got = append(got, ap)
		},
		func(string, ...any) {},
	)

	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		done := len(got) >= 2
		snapshot := append([]netip.AddrPort(nil), got...)
		mu.Unlock()
		if done {
			if snapshot[0] != a1 || snapshot[1] != a2 {
				t.Fatalf("updates = %v, want [%v %v] (once per distinct address)", snapshot, a1, a2)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("updates = %v, want two", snapshot)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
