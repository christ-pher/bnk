package server_test

import (
	"testing"

	"github.com/christ-pher/bnk/internal/netmap"
)

// Names key the ACL, so letting a second node claim an existing name
// lets it inherit that name's permissions as a traffic source and locks
// the real node out.
func TestEnrollRejectsADuplicateName(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "admin-laptop", ident32(t, 1).pub)

	code, body := e.enrollRaw(t, "admin-laptop", ident32(t, 2).pub, netmap.Key{9})
	if code == 200 {
		t.Fatal("a second node was allowed to claim an existing name")
	}
	t.Logf("rejected with %d: %s", code, body)
}

// Disco keys identify a peer's probe traffic, so a duplicate lets one
// node hijack another's path attribution on every third machine.
func TestEnrollRejectsADuplicateDiscoKey(t *testing.T) {
	e := startServer(t)
	shared := netmap.Key{7, 7, 7}
	if code, body := e.enrollRaw(t, "first", ident32(t, 1).pub, shared); code != 200 {
		t.Fatalf("first enrollment failed: %d %s", code, body)
	}
	code, body := e.enrollRaw(t, "second", ident32(t, 2).pub, shared)
	if code == 200 {
		t.Fatal("a second node was allowed to reuse a disco key")
	}
	t.Logf("rejected with %d: %s", code, body)
}

// Re-enrolling the same node key must still work: that is how a client
// recovers after losing its local state.
func TestEnrollAllowsTheSameNodeToReenroll(t *testing.T) {
	e := startServer(t)
	id := ident32(t, 1)
	first := e.enroll(t, "laptop", id.pub)
	again := e.enroll(t, "laptop", id.pub)
	if first.NodeID != again.NodeID || first.IP != again.IP {
		t.Errorf("re-enrolment changed identity: %v/%v then %v/%v",
			first.NodeID, first.IP, again.NodeID, again.IP)
	}
}
