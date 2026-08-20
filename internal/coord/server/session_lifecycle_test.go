package server_test

import (
	"testing"
	"time"
)

func TestNewSessionReplacesOldAndOldDoneFires(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "alpha", key32(1))
	first, _ := dialSession(t, e, key32(1))
	dialSession(t, e, key32(1))

	select {
	case <-first.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("Done() did not fire after a new session replaced this one")
	}
}
