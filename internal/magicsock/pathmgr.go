package magicsock

import (
	"crypto/rand"
	"fmt"
	"net/netip"
	"sync"
	"time"

	"vpnmesh/internal/disco"
)

const (
	keepaliveEvery = 5 * time.Second  // ping the proven path this often
	demoteAfter    = 15 * time.Second // silence on a proven path → relay
	reprobeEvery   = 15 * time.Second // retry discovery while on relay
	pingExpiry     = 10 * time.Second // forget unanswered pings
)

// PathManagerConfig injects all I/O and time, keeping the state machine
// pure enough to unit-test every transition.
type PathManagerConfig struct {
	DiscoPriv [32]byte
	DiscoPub  [32]byte
	Clock     func() time.Time
	SendRaw   func(to netip.AddrPort, pkt []byte) error // UDP via the Bind
	SendFwd   func(peer NodeKey, payload []byte) error  // disco_fwd via coord
	SetAddr   func(peer NodeKey, addr netip.AddrPort)   // proven path → Bind
	ClearAddr func(peer NodeKey)                        // demotion → Bind
	// AddAddrHint registers a candidate as a valid inbound source for the
	// peer; the peer may send from any candidate, not just the proven one.
	AddAddrHint func(peer NodeKey, addr netip.AddrPort)
}

type sentPing struct {
	peer NodeKey
	to   netip.AddrPort
	at   time.Time
}

type peerState struct {
	discoKey   [32]byte
	candidates map[netip.AddrPort]bool
	best       netip.AddrPort // zero = relay
	lastPong   time.Time
	lastPing   time.Time
	lastProbe  time.Time
}

// PathManager runs per-peer path discovery: call-me-maybe advertisement,
// simultaneous ping spray (the hole punch), pong-proven path selection,
// keepalive, and demotion back to relay on silence.
type PathManager struct {
	cfg PathManagerConfig

	mu         sync.Mutex
	meshPrefix netip.Prefix
	peers      map[NodeKey]*peerState
	byDiscoKey map[[32]byte]NodeKey
	pings      map[[12]byte]sentPing
	waiters    map[[12]byte]chan PingResult // Ping() calls awaiting pongs
	selfEps    []netip.AddrPort
}

func NewPathManager(cfg PathManagerConfig) *PathManager {
	return &PathManager{
		cfg:        cfg,
		peers:      make(map[NodeKey]*peerState),
		byDiscoKey: make(map[[32]byte]NodeKey),
		pings:      make(map[[12]byte]sentPing),
		waiters:    make(map[[12]byte]chan PingResult),
	}
}

// SetMeshPrefix declares the tunnel network. Any candidate inside it is
// refused: probing a tunnel address routes through the tunnel itself and
// proves a looping path that then swallows all traffic.
func (pm *PathManager) SetMeshPrefix(p netip.Prefix) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.meshPrefix = p
}

// badCandidateLocked reports whether addr must never be probed; pm.mu
// must be held.
func (pm *PathManager) badCandidateLocked(addr netip.AddrPort) bool {
	return pm.meshPrefix.IsValid() && pm.meshPrefix.Contains(addr.Addr())
}

func (pm *PathManager) SetSelfEndpoints(eps []netip.AddrPort) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.selfEps = append([]netip.AddrPort(nil), eps...)
}

// SetPeer registers or updates a peer. New candidates merge in; a proven
// path survives updates (liveness is Tick's business, not the netmap's).
func (pm *PathManager) SetPeer(key NodeKey, discoKey [32]byte, cands []netip.AddrPort) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	ps, ok := pm.peers[key]
	if !ok || ps.discoKey != discoKey {
		ps = &peerState{discoKey: discoKey, candidates: make(map[netip.AddrPort]bool)}
		pm.peers[key] = ps
	}
	pm.byDiscoKey[discoKey] = key
	for _, c := range cands {
		if pm.badCandidateLocked(c) {
			continue
		}
		ps.candidates[c] = true
		if pm.cfg.AddAddrHint != nil {
			pm.cfg.AddAddrHint(key, c)
		}
	}
}

func (pm *PathManager) RemovePeer(key NodeKey) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if ps, ok := pm.peers[key]; ok {
		delete(pm.byDiscoKey, ps.discoKey)
		delete(pm.peers, key)
	}
}

// TriggerProbe starts a discovery round: advertise our endpoints via the
// server and ping every known candidate.
func (pm *PathManager) TriggerProbe(key NodeKey) {
	pm.mu.Lock()
	ps, ok := pm.peers[key]
	if !ok {
		pm.mu.Unlock()
		return
	}
	ps.lastProbe = pm.cfg.Clock()
	selfEps := append([]netip.AddrPort(nil), pm.selfEps...)
	cands := candidateList(ps)
	discoKey := ps.discoKey
	pm.mu.Unlock()

	if len(selfEps) > 0 {
		cmm := disco.Seal(disco.CallMeMaybe{Endpoints: selfEps}, pm.cfg.DiscoPriv, pm.cfg.DiscoPub, discoKey)
		pm.cfg.SendFwd(key, cmm)
	}
	pm.pingAll(key, discoKey, cands)
}

func candidateList(ps *peerState) []netip.AddrPort {
	out := make([]netip.AddrPort, 0, len(ps.candidates))
	for c := range ps.candidates {
		out = append(out, c)
	}
	return out
}

// pingAll sends a sealed ping to each address, recording txids so pongs
// can prove the path.
func (pm *PathManager) pingAll(key NodeKey, discoKey [32]byte, addrs []netip.AddrPort) {
	now := pm.cfg.Clock()
	for _, addr := range addrs {
		var txid [12]byte
		rand.Read(txid[:])
		pm.mu.Lock()
		pm.pings[txid] = sentPing{peer: key, to: addr, at: now}
		if ps, ok := pm.peers[key]; ok {
			ps.lastPing = now
		}
		pm.mu.Unlock()
		pkt := disco.Seal(disco.Ping{TxID: txid}, pm.cfg.DiscoPriv, pm.cfg.DiscoPub, discoKey)
		pm.cfg.SendRaw(addr, pkt)
	}
}

// HandleDisco processes a disco packet that arrived over UDP from src.
func (pm *PathManager) HandleDisco(src netip.AddrPort, pkt []byte) {
	sender, msg, err := disco.Open(pkt, pm.cfg.DiscoPriv)
	if err != nil {
		return
	}
	pm.mu.Lock()
	key, known := pm.byDiscoKey[sender]
	ps := pm.peers[key]
	pm.mu.Unlock()
	if !known || ps == nil {
		return // decryption is not authentication: sender must be a known peer
	}

	switch m := msg.(type) {
	case disco.Ping:
		// A ping proves the peer can reach us at this path; answer with
		// their observed address and remember theirs as a candidate —
		// unless it arrived through the tunnel itself.
		pm.mu.Lock()
		if !pm.badCandidateLocked(src) {
			ps.candidates[src] = true
			pm.mu.Unlock()
			if pm.cfg.AddAddrHint != nil {
				pm.cfg.AddAddrHint(key, src)
			}
		} else {
			pm.mu.Unlock()
		}
		pong := disco.Seal(disco.Pong{TxID: m.TxID, Observed: src}, pm.cfg.DiscoPriv, pm.cfg.DiscoPub, ps.discoKey)
		pm.cfg.SendRaw(src, pong)
	case disco.Pong:
		pm.mu.Lock()
		sp, outstanding := pm.pings[m.TxID]
		if outstanding {
			delete(pm.pings, m.TxID)
		}
		if !outstanding || sp.peer != key {
			pm.mu.Unlock()
			return
		}
		now := pm.cfg.Clock()
		ps.lastPong = now
		promote := ps.best != sp.to
		ps.best = sp.to
		if w, waiting := pm.waiters[m.TxID]; waiting {
			select {
			case w <- PingResult{Addr: sp.to, RTT: now.Sub(sp.at)}:
			default:
			}
			delete(pm.waiters, m.TxID)
		}
		pm.mu.Unlock()
		if promote {
			pm.cfg.SetAddr(key, sp.to)
		}
	case disco.CallMeMaybe:
		pm.punch(key, ps, m.Endpoints)
	}
}

// HandleDiscoFwd processes a sealed disco payload forwarded by the server
// (only call-me-maybe travels this channel).
func (pm *PathManager) HandleDiscoFwd(payload []byte) {
	sender, msg, err := disco.Open(payload, pm.cfg.DiscoPriv)
	if err != nil {
		return
	}
	pm.mu.Lock()
	key, known := pm.byDiscoKey[sender]
	ps := pm.peers[key]
	pm.mu.Unlock()
	if !known || ps == nil {
		return
	}
	if cmm, ok := msg.(disco.CallMeMaybe); ok {
		pm.punch(key, ps, cmm.Endpoints)
	}
}

// punch merges freshly advertised endpoints and pings them all at
// once — the simultaneous transmit that opens NAT mappings on both sides.
func (pm *PathManager) punch(key NodeKey, ps *peerState, eps []netip.AddrPort) {
	pm.mu.Lock()
	kept := eps[:0]
	for _, ep := range eps {
		if pm.badCandidateLocked(ep) {
			continue
		}
		ps.candidates[ep] = true
		kept = append(kept, ep)
	}
	cands := candidateList(ps)
	discoKey := ps.discoKey
	pm.mu.Unlock()
	if pm.cfg.AddAddrHint != nil {
		for _, ep := range kept {
			pm.cfg.AddAddrHint(key, ep)
		}
	}
	pm.pingAll(key, discoKey, cands)
}

// PeerDebug is a diagnostic snapshot of one peer's path state.
type PeerDebug struct {
	Best        netip.AddrPort
	LastPongAge time.Duration // since the last proof of life (0 if never)
	LastPingAge time.Duration
	Candidates  []netip.AddrPort
}

func (pm *PathManager) PeerDebug(key NodeKey) (PeerDebug, bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	ps, ok := pm.peers[key]
	if !ok {
		return PeerDebug{}, false
	}
	now := pm.cfg.Clock()
	d := PeerDebug{Best: ps.best, Candidates: candidateList(ps)}
	if !ps.lastPong.IsZero() {
		d.LastPongAge = now.Sub(ps.lastPong)
	}
	if !ps.lastPing.IsZero() {
		d.LastPingAge = now.Sub(ps.lastPing)
	}
	return d, true
}

// PingResult reports a successful disco ping round trip.
type PingResult struct {
	Addr netip.AddrPort
	RTT  time.Duration
}

// Ping probes the peer's candidate paths and returns the first proven one
// with its round-trip time. Used by the vpn ping diagnostic.
func (pm *PathManager) Ping(key NodeKey, timeout time.Duration) (PingResult, error) {
	pm.mu.Lock()
	ps, ok := pm.peers[key]
	if !ok {
		pm.mu.Unlock()
		return PingResult{}, fmt.Errorf("magicsock: unknown peer %s", key)
	}
	targets := candidateList(ps)
	if ps.best.IsValid() {
		targets = append([]netip.AddrPort{ps.best}, targets...)
	}
	discoKey := ps.discoKey
	pm.mu.Unlock()
	if len(targets) == 0 {
		return PingResult{}, fmt.Errorf("magicsock: no candidate paths for %s", key)
	}

	ch := make(chan PingResult, 1)
	now := pm.cfg.Clock()
	var txids [][12]byte
	for _, addr := range targets {
		var txid [12]byte
		rand.Read(txid[:])
		pm.mu.Lock()
		pm.pings[txid] = sentPing{peer: key, to: addr, at: now}
		pm.waiters[txid] = ch
		pm.mu.Unlock()
		txids = append(txids, txid)
		pm.cfg.SendRaw(addr, disco.Seal(disco.Ping{TxID: txid}, pm.cfg.DiscoPriv, pm.cfg.DiscoPub, discoKey))
	}
	defer func() {
		pm.mu.Lock()
		for _, txid := range txids {
			delete(pm.waiters, txid)
		}
		pm.mu.Unlock()
	}()

	select {
	case res := <-ch:
		return res, nil
	case <-time.After(timeout):
		return PingResult{}, fmt.Errorf("magicsock: ping to %s timed out after %v", key, timeout)
	}
}

// Tick advances timers: keepalives on proven paths, demotion after
// silence, periodic reprobe while relayed, and ping expiry.
func (pm *PathManager) Tick() {
	now := pm.cfg.Clock()

	pm.mu.Lock()
	for txid, sp := range pm.pings {
		if now.Sub(sp.at) > pingExpiry {
			delete(pm.pings, txid)
		}
	}
	type action struct {
		key      NodeKey
		discoKey [32]byte
		ping     netip.AddrPort // keepalive target (zero = none)
		demote   bool
		reprobe  bool
	}
	var acts []action
	for key, ps := range pm.peers {
		a := action{key: key, discoKey: ps.discoKey}
		if ps.best.IsValid() {
			if now.Sub(ps.lastPong) > demoteAfter {
				a.demote = true
				ps.best = netip.AddrPort{}
				ps.lastProbe = now
				a.reprobe = true
			} else if now.Sub(ps.lastPing) >= keepaliveEvery {
				a.ping = ps.best
			}
		} else if now.Sub(ps.lastProbe) >= reprobeEvery {
			ps.lastProbe = now
			a.reprobe = true
		}
		if a.demote || a.reprobe || a.ping.IsValid() {
			acts = append(acts, a)
		}
	}
	pm.mu.Unlock()

	for _, a := range acts {
		if a.demote {
			pm.cfg.ClearAddr(a.key)
		}
		if a.ping.IsValid() {
			pm.pingAll(a.key, a.discoKey, []netip.AddrPort{a.ping})
		}
		if a.reprobe {
			pm.TriggerProbe(a.key)
		}
	}
}
