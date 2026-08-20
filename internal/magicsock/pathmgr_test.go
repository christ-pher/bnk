package magicsock

import (
	"net/netip"
	"sync"
	"testing"
	"time"

	"vpnmesh/internal/disco"
)

// pmHarness wires a PathManager to captured I/O and a fake clock. The
// mutex matters only for tests that call PathManager from goroutines.
type pmHarness struct {
	t     *testing.T
	pm    *PathManager
	mu    sync.Mutex
	now   time.Time
	prv   [32]byte // our disco priv
	pub   [32]byte
	pPriv [32]byte // the peer's disco keypair
	pPub  [32]byte
	peer  NodeKey

	rawSent  []rawPkt // SendRaw calls (pings/pongs on the wire)
	fwdSent  [][]byte // disco_fwd payloads to the peer
	setAddrs []netip.AddrPort
	cleared  int
}

type rawPkt struct {
	to  netip.AddrPort
	pkt []byte
}

func newPMHarness(t *testing.T) *pmHarness {
	t.Helper()
	h := &pmHarness{t: t, now: time.Unix(1700000000, 0), peer: NodeKey{0xAA}}
	var err error
	h.prv, h.pub, err = disco.NewKeypair()
	if err != nil {
		t.Fatal(err)
	}
	h.pPriv, h.pPub, err = disco.NewKeypair()
	if err != nil {
		t.Fatal(err)
	}
	h.pm = NewPathManager(PathManagerConfig{
		DiscoPriv: h.prv,
		DiscoPub:  h.pub,
		Clock: func() time.Time {
			h.mu.Lock()
			defer h.mu.Unlock()
			return h.now
		},
		SendRaw: func(to netip.AddrPort, pkt []byte) error {
			h.mu.Lock()
			defer h.mu.Unlock()
			h.rawSent = append(h.rawSent, rawPkt{to, append([]byte(nil), pkt...)})
			return nil
		},
		SendFwd: func(peer NodeKey, payload []byte) error {
			if peer != h.peer {
				t.Errorf("fwd to wrong peer %v", peer)
			}
			h.mu.Lock()
			defer h.mu.Unlock()
			h.fwdSent = append(h.fwdSent, append([]byte(nil), payload...))
			return nil
		},
		SetAddr: func(peer NodeKey, addr netip.AddrPort) {
			h.mu.Lock()
			defer h.mu.Unlock()
			h.setAddrs = append(h.setAddrs, addr)
		},
		ClearAddr: func(peer NodeKey) {
			h.mu.Lock()
			defer h.mu.Unlock()
			h.cleared++
		},
	})
	return h
}

// raw returns a snapshot of SendRaw calls, safe from any goroutine.
func (h *pmHarness) raw() []rawPkt {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]rawPkt(nil), h.rawSent...)
}

func (h *pmHarness) advance(d time.Duration) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.now = h.now.Add(d)
}

func (h *pmHarness) openAll(pkts []rawPkt) []disco.Message {
	h.t.Helper()
	var out []disco.Message
	for _, rp := range pkts {
		_, msg, err := disco.Open(rp.pkt, h.pPriv)
		if err != nil {
			h.t.Fatalf("peer could not open our disco packet: %v", err)
		}
		out = append(out, msg)
	}
	return out
}

var (
	cand1 = netip.MustParseAddrPort("192.0.2.1:41641")
	cand2 = netip.MustParseAddrPort("198.51.100.2:51820")
)

func (h *pmHarness) setPeerAndProbe() {
	h.pm.SetSelfEndpoints([]netip.AddrPort{netip.MustParseAddrPort("203.0.113.9:41641")})
	h.pm.SetPeer(h.peer, h.pPub, []netip.AddrPort{cand1, cand2})
	h.pm.TriggerProbe(h.peer)
}

func TestTriggerProbeSendsCallMeMaybeAndPingsCandidates(t *testing.T) {
	h := newPMHarness(t)
	h.setPeerAndProbe()

	if len(h.fwdSent) != 1 {
		t.Fatalf("fwd sent %d payloads, want 1 call-me-maybe", len(h.fwdSent))
	}
	_, msg, err := disco.Open(h.fwdSent[0], h.pPriv)
	if err != nil {
		t.Fatal(err)
	}
	cmm, ok := msg.(disco.CallMeMaybe)
	if !ok || len(cmm.Endpoints) != 1 || cmm.Endpoints[0] != netip.MustParseAddrPort("203.0.113.9:41641") {
		t.Errorf("cmm = %#v, want our self endpoint", msg)
	}

	pings := h.openAll(h.rawSent)
	if len(pings) != 2 {
		t.Fatalf("sent %d raw packets, want pings to both candidates", len(pings))
	}
	tos := map[netip.AddrPort]bool{h.rawSent[0].to: true, h.rawSent[1].to: true}
	if !tos[cand1] || !tos[cand2] {
		t.Errorf("pings went to %v, want %v and %v", tos, cand1, cand2)
	}
}

func TestPongProvesPathAndSetsAddr(t *testing.T) {
	h := newPMHarness(t)
	h.setPeerAndProbe()

	// The peer answers the ping that went to cand1.
	var txid [12]byte
	for i, rp := range h.rawSent {
		if rp.to == cand1 {
			txid = h.openAll(h.rawSent[i : i+1])[0].(disco.Ping).TxID
		}
	}
	pong := disco.Seal(disco.Pong{TxID: txid, Observed: netip.MustParseAddrPort("203.0.113.9:41641")}, h.pPriv, h.pPub, h.pub)
	h.pm.HandleDisco(cand1, pong)

	if len(h.setAddrs) != 1 || h.setAddrs[0] != cand1 {
		t.Fatalf("SetAddr calls = %v, want [%v]", h.setAddrs, cand1)
	}
}

func TestInboundPingGetsPongWithObservedAddr(t *testing.T) {
	h := newPMHarness(t)
	h.pm.SetPeer(h.peer, h.pPub, nil)

	from := netip.MustParseAddrPort("198.51.100.7:9999")
	ping := disco.Seal(disco.Ping{TxID: [12]byte{4}}, h.pPriv, h.pPub, h.pub)
	h.pm.HandleDisco(from, ping)

	if len(h.rawSent) != 1 || h.rawSent[0].to != from {
		t.Fatalf("raw sent = %v, want one pong back to %v", h.rawSent, from)
	}
	pong := h.openAll(h.rawSent)[0].(disco.Pong)
	if pong.TxID != [12]byte{4} || pong.Observed != from {
		t.Errorf("pong = %#v, want txid 4 observed %v", pong, from)
	}
}

func TestCallMeMaybeTriggersImmediatePings(t *testing.T) {
	h := newPMHarness(t)
	h.pm.SetPeer(h.peer, h.pPub, nil)

	cmm := disco.Seal(disco.CallMeMaybe{Endpoints: []netip.AddrPort{cand1, cand2}}, h.pPriv, h.pPub, h.pub)
	h.pm.HandleDiscoFwd(cmm)

	msgs := h.openAll(h.rawSent)
	if len(msgs) != 2 {
		t.Fatalf("sent %d packets after call-me-maybe, want 2 pings (the punch)", len(msgs))
	}
	for _, m := range msgs {
		if _, ok := m.(disco.Ping); !ok {
			t.Errorf("sent %#v, want pings", m)
		}
	}
}

func TestKeepaliveThenDemotionOnSilence(t *testing.T) {
	h := newPMHarness(t)
	h.setPeerAndProbe()
	var txid [12]byte
	for i, rp := range h.rawSent {
		if rp.to == cand1 {
			txid = h.openAll(h.rawSent[i : i+1])[0].(disco.Ping).TxID
		}
	}
	h.pm.HandleDisco(cand1, disco.Seal(disco.Pong{TxID: txid, Observed: cand1}, h.pPriv, h.pPub, h.pub))
	h.rawSent = nil

	// 6s later a keepalive ping goes to the proven path.
	h.now = h.now.Add(6 * time.Second)
	h.pm.Tick()
	if len(h.rawSent) == 0 || h.rawSent[0].to != cand1 {
		t.Fatalf("no keepalive ping to proven path after 6s: %v", h.rawSent)
	}

	// 20s of silence: the path is demoted and rediscovery starts.
	h.now = h.now.Add(20 * time.Second)
	h.pm.Tick()
	if h.cleared != 1 {
		t.Errorf("ClearAddr calls = %d, want 1 (demoted after silence)", h.cleared)
	}
}

func TestUnknownDiscoKeyIsDropped(t *testing.T) {
	h := newPMHarness(t)
	h.pm.SetPeer(h.peer, h.pPub, nil)

	strangerPriv, strangerPub, err := disco.NewKeypair()
	if err != nil {
		t.Fatal(err)
	}
	ping := disco.Seal(disco.Ping{TxID: [12]byte{6}}, strangerPriv, strangerPub, h.pub)
	h.pm.HandleDisco(netip.MustParseAddrPort("192.0.2.66:1"), ping)

	if len(h.rawSent) != 0 || len(h.setAddrs) != 0 {
		t.Error("path manager reacted to a stranger's disco packet")
	}
}
