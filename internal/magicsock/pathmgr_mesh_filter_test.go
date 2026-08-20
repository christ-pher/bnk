package magicsock

import (
	"net/netip"
	"testing"

	"vpnmesh/internal/disco"
)

// The path manager must refuse candidates inside the mesh prefix, from
// every source: the netmap, call-me-maybe advertisements, and observed
// ping sources. A tunnel address as a candidate creates a self-loop.
func TestMeshAddressesAreNeverCandidates(t *testing.T) {
	h := newPMHarness(t)
	h.pm.SetMeshPrefix(netip.MustParsePrefix("100.64.0.0/10"))
	tunnelEp := netip.MustParseAddrPort("100.64.0.2:41641")

	// Via netmap:
	h.pm.SetSelfEndpoints([]netip.AddrPort{netip.MustParseAddrPort("203.0.113.9:1")})
	h.pm.SetPeer(h.peer, h.pPub, []netip.AddrPort{tunnelEp, cand1})
	h.pm.TriggerProbe(h.peer)
	for _, rp := range h.raw() {
		if rp.to == tunnelEp {
			t.Fatal("probed a mesh (tunnel) address from netmap candidates")
		}
	}

	// Via call-me-maybe:
	cmm := disco.Seal(disco.CallMeMaybe{Endpoints: []netip.AddrPort{tunnelEp}}, h.pPriv, h.pPub, h.pub)
	h.pm.HandleDiscoFwd(cmm)
	for _, rp := range h.raw() {
		if rp.to == tunnelEp {
			t.Fatal("probed a mesh address advertised via call-me-maybe")
		}
	}

	// Via observed ping source (a ping that arrived through the tunnel):
	ping := disco.Seal(disco.Ping{TxID: [12]byte{1}}, h.pPriv, h.pPub, h.pub)
	h.pm.HandleDisco(tunnelEp, ping)
	d, _ := h.pm.PeerDebug(h.peer)
	for _, c := range d.Candidates {
		if c == tunnelEp {
			t.Fatal("mesh address adopted as candidate from ping source")
		}
	}
}
