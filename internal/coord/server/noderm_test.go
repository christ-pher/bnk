package server_test

import (
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/netmap"
)

func nodeNames(t *testing.T, e *env) map[string]bool {
	t.Helper()
	out := map[string]bool{}
	for _, n := range adminNodes(t, e) {
		out[n.Name] = true
	}
	return out
}

func TestRemoveNodeDropsItFromTheMesh(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "alpha", ident32(t, 1).pub)
	e.enroll(t, "beta", ident32(t, 2).pub)

	if err := e.srv.RemoveNode("beta"); err != nil {
		t.Fatal(err)
	}
	names := nodeNames(t, e)
	if names["beta"] {
		t.Error("beta is still registered after removal")
	}
	if !names["alpha"] {
		t.Error("alpha was removed too")
	}
}

func TestRemoveNodeIsAnErrorForUnknownNames(t *testing.T) {
	e := startServer(t)
	if err := e.srv.RemoveNode("ghost"); err == nil {
		t.Error("expected an error removing a node that does not exist")
	}
}

// A removed node's address must go back into the pool.
func TestRemovedNodeAddressIsReused(t *testing.T) {
	e := startServer(t)
	first := e.enroll(t, "alpha", ident32(t, 1).pub)
	if err := e.srv.RemoveNode("alpha"); err != nil {
		t.Fatal(err)
	}
	again := e.enroll(t, "gamma", ident32(t, 3).pub)
	if again.IP != first.IP {
		t.Errorf("new node got %v, want the freed address %v", again.IP, first.IP)
	}
}

// Peers must stop seeing a removed node without reconnecting.
func TestRemoveNodeBroadcastsToPeers(t *testing.T) {
	e := startServer(t)
	alpha := ident32(t, 1)
	e.enroll(t, "alpha", alpha.pub)
	e.enroll(t, "beta", ident32(t, 2).pub)

	_, nms := dialSession(t, e, alpha.priv)
	waitNetmap(t, nms, func(nm netmap.Netmap) bool { return len(nm.Peers) == 1 })

	if err := e.srv.RemoveNode("beta"); err != nil {
		t.Fatal(err)
	}
	waitNetmap(t, nms, func(nm netmap.Netmap) bool { return len(nm.Peers) == 0 })
}

// A node that removes itself over its authenticated session is gone for
// good: that is what the client's `bnk leave` uses.
func TestSessionLeaveRemovesTheNode(t *testing.T) {
	e := startServer(t)
	alpha := ident32(t, 1)
	e.enroll(t, "alpha", alpha.pub)
	e.enroll(t, "beta", ident32(t, 2).pub)

	sess, _ := dialSession(t, e, alpha.priv)
	if err := sess.Leave(); err != nil {
		t.Fatal(err)
	}

	deadline := time.Now().Add(5 * time.Second)
	for {
		if !nodeNames(t, e)["alpha"] {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("alpha is still registered after leaving")
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !nodeNames(t, e)["beta"] {
		t.Error("leaving removed the wrong node")
	}
}
