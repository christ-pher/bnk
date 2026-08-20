package dataplane_test

// Phase 2 path-selection v0: a peer that advertises an endpoint we can
// never handshake with (e.g. its LAN address, unreachable across NATs)
// must be demoted to the relay path instead of staying broken.

import (
	"context"
	"net/http/httptest"
	"net/netip"
	"path/filepath"
	"testing"
	"time"

	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/store"
)

func TestBogusDirectEndpointFallsBackToRelay(t *testing.T) {
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

	a := startRelayOnlyClient(t, ts.URL, enrollKey, "alpha")
	b := startRelayOnlyClient(t, ts.URL, enrollKey, "beta")

	// Both advertise an unreachable endpoint (port 9, nothing listening),
	// so the direct path can never handshake.
	bogus := []netip.AddrPort{netip.MustParseAddrPort("127.0.0.1:9")}
	if err := a.sess.SendEndpoints(bogus); err != nil {
		t.Fatal(err)
	}
	if err := b.sess.SendEndpoints(bogus); err != nil {
		t.Fatal(err)
	}

	ln, err := b.net.ListenTCPAddrPort(netip.AddrPortFrom(b.ip, 9999))
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
			go func() {
				defer c.Close()
				buf := make([]byte, 64)
				if n, err := c.Read(buf); err == nil {
					c.Write(buf[:n])
				}
			}()
		}
	}()

	deadline := time.Now().Add(40 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		c, err := a.net.DialContextTCPAddrPort(ctx, netip.AddrPortFrom(b.ip, 9999))
		cancel()
		if err == nil {
			c.Close()
			return // tunnel came up despite the bogus endpoints
		}
		if time.Now().After(deadline) {
			t.Fatalf("tunnel never fell back to relay: %v", err)
		}
	}
}
