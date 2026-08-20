package server_test

import (
	"bytes"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/coord/client"
	"github.com/christ-pher/bnk/internal/netmap"
)

type fwd struct {
	src     netmap.NodeID
	payload []byte
}

func dialWithDisco(t *testing.T, e *env, priv netmap.Key) (*client.Session, chan fwd, chan netmap.Netmap) {
	t.Helper()
	ch := make(chan fwd, 16)
	nms := make(chan netmap.Netmap, 16)
	sess, err := client.Dial(t.Context(), e.ts.URL, nil, priv, client.Handlers{
		OnNetmap: func(nm netmap.Netmap) { nms <- nm },
		OnDiscoFwd: func(src netmap.NodeID, payload []byte) {
			ch <- fwd{src, append([]byte(nil), payload...)}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sess.Close)
	return sess, ch, nms
}

func TestDiscoFwdForwardedWithSourceStamped(t *testing.T) {
	e := startServer(t)
	idA, idB := ident32(t, 1), ident32(t, 2)
	a := e.enroll(t, "alpha", idA.pub)
	b := e.enroll(t, "beta", idB.pub)

	sessA, _, nmsA := dialWithDisco(t, e, idA.priv)
	_, gotB, _ := dialWithDisco(t, e, idB.priv)

	waitNetmap(t, nmsA, func(nm netmap.Netmap) bool {
		return len(nm.Peers) == 1 && nm.Peers[0].Online
	})

	if err := sessA.SendDiscoFwd(b.NodeID, []byte("sealed-cmm")); err != nil {
		t.Fatal(err)
	}

	select {
	case f := <-gotB:
		if f.src != a.NodeID {
			t.Errorf("src = %d, want %d (server must stamp true source)", f.src, a.NodeID)
		}
		if !bytes.Equal(f.payload, []byte("sealed-cmm")) {
			t.Errorf("payload = %q", f.payload)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("disco_fwd never arrived")
	}
}
