package server_test

import (
	"testing"
	"time"
)

func TestNewSessionReplacesOldAndOldDoneFires(t *testing.T) {
	e := startServer(t)
	id := ident32(t, 1)
	e.enroll(t, "alpha", id.pub)
	first, _ := dialSession(t, e, id.priv)
	dialSession(t, e, id.priv)

	select {
	case <-first.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("Done() did not fire after a new session replaced this one")
	}
}
