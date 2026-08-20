package magicsock

import (
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/disco"
)

func TestPeerDebugReportsPathState(t *testing.T) {
	h := newPMHarness(t)
	h.setPeerAndProbe()

	d, ok := h.pm.PeerDebug(h.peer)
	if !ok {
		t.Fatal("PeerDebug: unknown peer")
	}
	if d.Best.IsValid() {
		t.Errorf("best = %v before any pong", d.Best)
	}
	if len(d.Candidates) != 2 {
		t.Errorf("candidates = %v, want the two seeded ones", d.Candidates)
	}

	// Prove cand1, then debug must show it with a fresh pong age.
	var txid [12]byte
	for _, rp := range h.raw() {
		if rp.to == cand1 {
			txid = h.openAll([]rawPkt{rp})[0].(disco.Ping).TxID
		}
	}
	h.advance(2 * time.Second)
	h.pm.HandleDisco(cand1, disco.Seal(disco.Pong{TxID: txid, Observed: cand1}, h.pPriv, h.pPub, h.pub))
	h.advance(3 * time.Second)

	d, _ = h.pm.PeerDebug(h.peer)
	if d.Best != cand1 {
		t.Errorf("best = %v, want %v", d.Best, cand1)
	}
	if d.LastPongAge != 3*time.Second {
		t.Errorf("lastPongAge = %v, want 3s", d.LastPongAge)
	}
}
