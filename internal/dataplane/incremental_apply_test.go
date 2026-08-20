package dataplane_test

// Netmap pushes happen routinely (endpoint refresh, nodes joining); they
// must not disturb established WireGuard sessions. The old replace_peers
// approach wiped handshake state on every push.

import (
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/store"
)

func TestNetmapPushPreservesHandshakeState(t *testing.T) {
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
	echoTCPOverTunnel(t, a, b)

	handshaked := func() bool {
		for _, pp := range a.engine.PeerPaths() {
			if !pp.LastHandshake.IsZero() {
				return true
			}
		}
		return false
	}
	if !handshaked() {
		t.Fatal("no handshake after traffic")
	}

	// A routine netmap push (e.g. an endpoint refresh) must not reset the
	// established session. Trigger several and give them time to apply.
	for i := 0; i < 3; i++ {
		if err := a.sess.SendEndpoints(nil); err != nil {
			t.Fatal(err)
		}
		time.Sleep(300 * time.Millisecond)
		if !handshaked() {
			t.Fatal("handshake state wiped by a netmap push")
		}
	}
}
