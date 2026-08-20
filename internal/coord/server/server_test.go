package server_test

import (
	"context"
	"net/http/httptest"
	"net/netip"
	"path/filepath"
	"testing"
	"time"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/coord/client"
	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/store"
)

type env struct {
	srv       *server.Server
	ts        *httptest.Server
	statePath string
	enrollKey string
}

func startServer(t *testing.T) *env {
	t.Helper()
	statePath := filepath.Join(t.TempDir(), "state.json")
	srv, err := server.New(store.NewFileStore(statePath))
	if err != nil {
		t.Fatal(err)
	}
	key, err := srv.NewEnrollKey()
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return &env{srv: srv, ts: ts, statePath: statePath, enrollKey: key}
}

func key32(b byte) netmap.Key {
	var k netmap.Key
	k[0] = b
	return k
}

func (e *env) enroll(t *testing.T, name string, nodeKey netmap.Key) coord.EnrollResponse {
	t.Helper()
	resp, err := client.Enroll(context.Background(), e.ts.URL, e.ts.Client(), coord.EnrollRequest{
		EnrollKey: e.enrollKey,
		Hostname:  name,
		OS:        "linux",
		NodeKey:   nodeKey,
		DiscoKey:  key32(0xD0),
	})
	if err != nil {
		t.Fatalf("enroll %s: %v", name, err)
	}
	return resp
}

func TestEnrollAssignsDistinctIPsAndPersists(t *testing.T) {
	e := startServer(t)
	a := e.enroll(t, "alpha", key32(1))
	b := e.enroll(t, "beta", key32(2))

	if a.IP == b.IP {
		t.Errorf("both nodes got IP %v", a.IP)
	}
	prefix := netip.MustParsePrefix("100.64.0.0/10")
	if !prefix.Contains(a.IP) || !prefix.Contains(b.IP) {
		t.Errorf("IPs %v %v outside %v", a.IP, b.IP, prefix)
	}

	st, err := store.NewFileStore(e.statePath).Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(st.Nodes) != 2 {
		t.Errorf("persisted %d nodes, want 2", len(st.Nodes))
	}
}

func TestEnrollRejectsBadKey(t *testing.T) {
	e := startServer(t)
	_, err := client.Enroll(context.Background(), e.ts.URL, e.ts.Client(), coord.EnrollRequest{
		EnrollKey: "wrong",
		Hostname:  "mallory",
		NodeKey:   key32(3),
	})
	if err == nil {
		t.Error("enroll with bad key succeeded, want error")
	}
}

func TestEnrollSameNodeKeyIsIdempotent(t *testing.T) {
	e := startServer(t)
	first := e.enroll(t, "alpha", key32(1))
	again := e.enroll(t, "alpha-reinstalled", key32(1))
	if first.NodeID != again.NodeID || first.IP != again.IP {
		t.Errorf("re-enroll got id=%d ip=%v, want id=%d ip=%v", again.NodeID, again.IP, first.NodeID, first.IP)
	}
}

// dialSession connects a client session and returns it plus a channel of
// received netmaps.
func dialSession(t *testing.T, e *env, nodeKey netmap.Key) (*client.Session, chan netmap.Netmap) {
	t.Helper()
	nms := make(chan netmap.Netmap, 16)
	sess, err := client.Dial(context.Background(), e.ts.URL, nil, nodeKey, client.Handlers{
		OnNetmap: func(nm netmap.Netmap) { nms <- nm },
	})
	if err != nil {
		t.Fatalf("dial session: %v", err)
	}
	t.Cleanup(sess.Close)
	return sess, nms
}

func waitNetmap(t *testing.T, ch chan netmap.Netmap, ok func(netmap.Netmap) bool) netmap.Netmap {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case nm := <-ch:
			if ok(nm) {
				return nm
			}
		case <-deadline:
			t.Fatal("timed out waiting for matching netmap")
		}
	}
}

func TestSessionReceivesNetmapListingPeers(t *testing.T) {
	e := startServer(t)
	a := e.enroll(t, "alpha", key32(1))
	b := e.enroll(t, "beta", key32(2))

	_, nms := dialSession(t, e, key32(1))
	nm := waitNetmap(t, nms, func(nm netmap.Netmap) bool { return len(nm.Peers) == 1 })
	if nm.SelfID != a.NodeID || nm.SelfIP.Addr() != a.IP {
		t.Errorf("self = %d %v, want %d %v", nm.SelfID, nm.SelfIP, a.NodeID, a.IP)
	}
	p := nm.Peers[0]
	if p.ID != b.NodeID || p.IP != b.IP || p.NodeKey != key32(2) || p.Name != "beta" {
		t.Errorf("peer = %+v, want beta id=%d ip=%v", p, b.NodeID, b.IP)
	}
}

func TestEndpointUpdatesReachOtherSessions(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "alpha", key32(1))
	e.enroll(t, "beta", key32(2))

	sessA, nmsA := dialSession(t, e, key32(1))
	_, nmsB := dialSession(t, e, key32(2))

	ep := netip.MustParseAddrPort("203.0.113.5:41641")
	if err := sessA.SendEndpoints([]netip.AddrPort{ep}); err != nil {
		t.Fatal(err)
	}

	nm := waitNetmap(t, nmsB, func(nm netmap.Netmap) bool {
		return len(nm.Peers) == 1 && len(nm.Peers[0].Endpoints) == 1
	})
	if nm.Peers[0].Endpoints[0] != ep {
		t.Errorf("peer endpoint = %v, want %v", nm.Peers[0].Endpoints[0], ep)
	}
	// A's own stream keeps flowing too (online-status pushes); drain lazily.
	select {
	case <-nmsA:
	default:
	}
}
