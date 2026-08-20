package magicsock

import (
	"net/netip"
	"testing"
	"time"

	"vpnmesh/internal/disco"
)

// Stale candidates (drifted NAT mappings, gone endpoints) must not be
// probed forever: the netmap is authoritative for advertised candidates,
// and learned ones (call-me-maybe, observed ping sources) expire.
func TestNetmapCandidatesArePruned(t *testing.T) {
	h := newPMHarness(t)
	h.pm.SetPeer(h.peer, h.pPub, []netip.AddrPort{cand1, cand2})
	h.pm.SetPeer(h.peer, h.pPub, []netip.AddrPort{cand1})

	d, _ := h.pm.PeerDebug(h.peer)
	if len(d.Candidates) != 1 || d.Candidates[0] != cand1 {
		t.Fatalf("candidates = %v, want only %v (netmap no longer lists %v)", d.Candidates, cand1, cand2)
	}
}

func TestLearnedCandidatesExpire(t *testing.T) {
	h := newPMHarness(t)
	h.pm.SetPeer(h.peer, h.pPub, []netip.AddrPort{cand1})

	learned := netip.MustParseAddrPort("198.51.100.99:4141")
	cmm := disco.Seal(disco.CallMeMaybe{Endpoints: []netip.AddrPort{learned}}, h.pPriv, h.pPub, h.pub)
	h.pm.HandleDiscoFwd(cmm)

	d, _ := h.pm.PeerDebug(h.peer)
	if len(d.Candidates) != 2 {
		t.Fatalf("candidates = %v, want netmap + learned", d.Candidates)
	}

	// Well past the learned TTL, a tick sweeps it; the netmap one stays.
	h.advance(10 * time.Minute)
	h.pm.Tick()
	d, _ = h.pm.PeerDebug(h.peer)
	if len(d.Candidates) != 1 || d.Candidates[0] != cand1 {
		t.Fatalf("candidates after expiry = %v, want only %v", d.Candidates, cand1)
	}
}
