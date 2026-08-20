package dataplane_test

// Phase 3 exit criterion: with a policy allowing only tcp/7777 to beta,
// alpha can reach beta:7777 but beta:8888 times out — enforced by beta's
// own userspace filter, pushed from the control server.

import (
	"context"
	"net/http/httptest"
	"net/netip"
	"path/filepath"
	"testing"
	"time"

	"vpnmesh/internal/acl"
	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/store"
)

func TestPolicyAllowsOnePortAndBlocksAnother(t *testing.T) {
	srv, err := server.New(store.NewFileStore(filepath.Join(t.TempDir(), "state.json")))
	if err != nil {
		t.Fatal(err)
	}
	enrollKey, err := srv.NewEnrollKey()
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	a := startClient(t, ts.URL, enrollKey, "alpha")
	b := startClient(t, ts.URL, enrollKey, "beta")

	if err := srv.SetPolicy(&acl.Policy{
		Rules: []acl.Rule{{From: []string{"alpha"}, To: []string{"beta"}, Allow: []string{"tcp/7777"}}},
	}); err != nil {
		t.Fatal(err)
	}

	for _, port := range []uint16{7777, 8888} {
		ln, err := b.net.ListenTCPAddrPort(netip.AddrPortFrom(b.ip, port))
		if err != nil {
			t.Fatal(err)
		}
		defer ln.Close()
		go func() {
			for {
				c, err := ln.Accept()
				if err != nil {
					return
				}
				c.Close()
			}
		}()
	}

	// The tunnel comes up before the policy push lands, so loop until the
	// steady state holds: allowed port connects AND denied port is blocked.
	tryDial := func(port uint16) bool {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		c, err := a.net.DialContextTCPAddrPort(ctx, netip.AddrPortFrom(b.ip, port))
		if err != nil {
			return false
		}
		c.Close()
		return true
	}
	deadline := time.Now().Add(30 * time.Second)
	for {
		if tryDial(7777) && !tryDial(8888) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("never reached policy steady state: 7777=%v 8888=%v (want true/false)",
				tryDial(7777), tryDial(8888))
		}
	}
}
