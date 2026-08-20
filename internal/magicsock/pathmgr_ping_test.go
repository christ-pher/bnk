package magicsock

import (
	"net/netip"
	"testing"
	"time"

	"vpnmesh/internal/disco"
)

func TestPingWaitsForPongAndReportsPathAndRTT(t *testing.T) {
	h := newPMHarness(t)
	h.setPeerAndProbe()
	h.mu.Lock()
	h.rawSent = nil
	h.mu.Unlock()

	type result struct {
		res PingResult
		err error
	}
	done := make(chan result, 1)
	go func() {
		res, err := h.pm.Ping(h.peer, 5*time.Second)
		done <- result{res, err}
	}()

	// Wait for the ping to hit the wire, then answer it from cand1.
	deadline := time.Now().Add(5 * time.Second)
	for len(h.raw()) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("Ping never sent anything")
		}
		time.Sleep(10 * time.Millisecond)
	}
	var answered bool
	for _, rp := range h.raw() {
		if rp.to == cand1 {
			txid := h.openAll([]rawPkt{rp})[0].(disco.Ping).TxID
			h.advance(30 * time.Millisecond)
			pong := disco.Seal(disco.Pong{TxID: txid, Observed: cand1}, h.pPriv, h.pPub, h.pub)
			h.pm.HandleDisco(cand1, pong)
			answered = true
		}
	}
	if !answered {
		t.Fatalf("no ping went to %v: %v", cand1, h.raw())
	}

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatal(r.err)
		}
		if r.res.Addr != cand1 {
			t.Errorf("addr = %v, want %v", r.res.Addr, cand1)
		}
		if r.res.RTT <= 0 {
			t.Errorf("rtt = %v, want > 0", r.res.RTT)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Ping never returned after pong")
	}
}

func TestPingTimesOutWhenNoPong(t *testing.T) {
	h := newPMHarness(t)
	h.pm.SetPeer(h.peer, h.pPub, []netip.AddrPort{cand1})

	if _, err := h.pm.Ping(h.peer, 50*time.Millisecond); err == nil {
		t.Error("Ping with no pong succeeded, want timeout error")
	}
}
