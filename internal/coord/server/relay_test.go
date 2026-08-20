package server_test

import (
	"bytes"
	"testing"
	"time"

	"vpnmesh/internal/coord/client"
	"vpnmesh/internal/netmap"
)

type relayed struct {
	src netmap.NodeID
	pkt []byte
}

func dialWithRelay(t *testing.T, e *env, nodeKey netmap.Key) (*client.Session, chan relayed, chan netmap.Netmap) {
	t.Helper()
	ch := make(chan relayed, 16)
	nms := make(chan netmap.Netmap, 16)
	sess, err := client.Dial(t.Context(), e.ts.URL, nil, nodeKey, client.Handlers{
		OnNetmap: func(nm netmap.Netmap) { nms <- nm },
		OnRelayData: func(src netmap.NodeID, pkt []byte) {
			ch <- relayed{src, append([]byte(nil), pkt...)}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sess.Close)
	return sess, ch, nms
}

func TestRelayForwardsWithSourceStampedByServer(t *testing.T) {
	e := startServer(t)
	a := e.enroll(t, "alpha", key32(1))
	b := e.enroll(t, "beta", key32(2))

	sessA, _, nmsA := dialWithRelay(t, e, key32(1))
	_, gotB, _ := dialWithRelay(t, e, key32(2))

	// Wait until the server sees beta online, else the frame is dropped.
	waitNetmap(t, nmsA, func(nm netmap.Netmap) bool {
		return len(nm.Peers) == 1 && nm.Peers[0].Online
	})

	if err := sessA.SendRelay(b.NodeID, []byte("wg-encrypted-bytes")); err != nil {
		t.Fatal(err)
	}

	select {
	case r := <-gotB:
		if r.src != a.NodeID {
			t.Errorf("src = %d, want alpha's id %d (server must stamp the true source)", r.src, a.NodeID)
		}
		if !bytes.Equal(r.pkt, []byte("wg-encrypted-bytes")) {
			t.Errorf("pkt = %q", r.pkt)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("relay frame never arrived")
	}
}

func TestRelayToOfflinePeerIsDroppedWithoutKillingSession(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "alpha", key32(1))
	b := e.enroll(t, "beta", key32(2))

	sessA, _, _ := dialWithRelay(t, e, key32(1))
	if err := sessA.SendRelay(b.NodeID, []byte("into the void")); err != nil {
		t.Fatal(err)
	}

	select {
	case <-sessA.Done():
		t.Fatal("session died after relaying to an offline peer")
	case <-time.After(300 * time.Millisecond):
	}
}
