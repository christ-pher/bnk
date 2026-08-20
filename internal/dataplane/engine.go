// Package dataplane assembles the client's packet path: TUN device,
// (later) ACL filter, WireGuard device, and magicsock Bind.
package dataplane

import (
	"context"
	"encoding/hex"
	"fmt"
	"net/netip"
	"strings"
	"sync"
	"time"

	"golang.zx2c4.com/wireguard/device"
	"golang.zx2c4.com/wireguard/tun"

	"vpnmesh/internal/filter"
	"vpnmesh/internal/magicsock"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/stunner"
)

// fallbackGrace is how long a never-handshaked peer may hold a direct
// address before being demoted to the relay path.
const fallbackGrace = 5 * time.Second

type Engine struct {
	mu          sync.Mutex
	bind        *magicsock.Bind
	dev         *device.Device
	filter      *filter.Filter
	pm          *magicsock.PathManager
	stun        *stunner.Client
	applied     map[magicsock.NodeKey]netip.AddrPort // blind direct addrs (legacy, no-disco peers)
	directSince map[magicsock.NodeKey]time.Time      // unproven blind paths
	pmDirect    map[magicsock.NodeKey]netip.AddrPort // disco-proven paths
	keyToID     map[magicsock.NodeKey]uint32
	fwdSend     func(dst uint32, payload []byte) error
	done        chan struct{}
}

// New brings up a WireGuard device on tunDev with a fresh magicsock Bind.
// The ACL filter sits between the device and the TUN; it starts in
// allow-all mode until a netmap carries a policy. The disco keypair feeds
// the path manager; peers that publish a disco key get proven paths (probe
// → punch → pong), others fall back to blind endpoints plus the watchdog.
func New(tunDev tun.Device, privateKey, discoPriv, discoPub [32]byte) (*Engine, error) {
	f := filter.New()
	bind := magicsock.NewBind()
	dev := device.NewDevice(filter.WrapTUN(tunDev, f), bind, device.NewLogger(device.LogLevelError, "wg: "))
	if err := dev.IpcSet(fmt.Sprintf("private_key=%s\n", hex.EncodeToString(privateKey[:]))); err != nil {
		dev.Close()
		return nil, err
	}
	if err := dev.Up(); err != nil {
		dev.Close()
		return nil, err
	}
	e := &Engine{
		bind:        bind,
		dev:         dev,
		filter:      f,
		applied:     make(map[magicsock.NodeKey]netip.AddrPort),
		directSince: make(map[magicsock.NodeKey]time.Time),
		pmDirect:    make(map[magicsock.NodeKey]netip.AddrPort),
		keyToID:     make(map[magicsock.NodeKey]uint32),
		done:        make(chan struct{}),
	}
	e.pm = magicsock.NewPathManager(magicsock.PathManagerConfig{
		DiscoPriv: discoPriv,
		DiscoPub:  discoPub,
		Clock:     time.Now,
		SendRaw:   bind.SendRaw,
		SendFwd: func(peer magicsock.NodeKey, payload []byte) error {
			e.mu.Lock()
			id, ok := e.keyToID[peer]
			send := e.fwdSend
			e.mu.Unlock()
			if !ok || send == nil {
				return nil
			}
			return send(id, payload)
		},
		SetAddr: func(peer magicsock.NodeKey, addr netip.AddrPort) {
			bind.SetPeerAddr(peer, addr)
			e.mu.Lock()
			e.pmDirect[peer] = addr
			e.mu.Unlock()
		},
		ClearAddr: func(peer magicsock.NodeKey) {
			bind.ClearPeerAddr(peer)
			e.mu.Lock()
			delete(e.pmDirect, peer)
			e.mu.Unlock()
		},
		AddAddrHint: bind.AddAddrHint,
	})
	bind.SetDiscoHandler(e.pm.HandleDisco)
	e.stun = stunner.NewClient(bind)
	go e.watchdog()
	return e, nil
}

// PeerDebug exposes the path manager's diagnostic snapshot for a peer.
func (e *Engine) PeerDebug(key magicsock.NodeKey) (magicsock.PeerDebug, bool) {
	return e.pm.PeerDebug(key)
}

// RelayStats reports WireGuard packets sent/received via the relay.
func (e *Engine) RelayStats() (tx, rx uint64) {
	return e.bind.RelayStats()
}

// PingPeer runs a disco ping round trip to the peer, proving (or timing
// out on) a direct path.
func (e *Engine) PingPeer(key magicsock.NodeKey, timeout time.Duration) (magicsock.PingResult, error) {
	return e.pm.Ping(key, timeout)
}

// QuerySTUN asks server for this node's reflexive address as seen from
// the WireGuard socket.
func (e *Engine) QuerySTUN(ctx context.Context, server netip.AddrPort) (netip.AddrPort, error) {
	return e.stun.Query(ctx, server)
}

// SetDiscoFwdSender wires the coordination session's disco_fwd transport
// into the path manager.
func (e *Engine) SetDiscoFwdSender(send func(dst uint32, payload []byte) error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.fwdSend = send
}

// HandleDiscoFwd feeds a server-forwarded disco payload to the path
// manager.
func (e *Engine) HandleDiscoFwd(payload []byte) {
	e.pm.HandleDiscoFwd(payload)
}

// SetMeshPrefix declares the tunnel network so the path manager never
// probes in-tunnel addresses (which would prove a self-looping path).
func (e *Engine) SetMeshPrefix(p netip.Prefix) {
	e.pm.SetMeshPrefix(p)
}

// SetSelfEndpoints tells the path manager which addresses to advertise in
// call-me-maybe messages.
func (e *Engine) SetSelfEndpoints(eps []netip.AddrPort) {
	e.pm.SetSelfEndpoints(eps)
}

// watchdog demotes direct paths that never produce a handshake, so a peer
// advertising an unreachable endpoint ends up on the relay instead of
// staying dark. Phase 4's path manager replaces this heuristic.
func (e *Engine) watchdog() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-e.done:
			return
		case <-ticker.C:
		}
		e.pm.Tick()
		shaken := e.handshakedPeers()
		e.mu.Lock()
		for key, since := range e.directSince {
			if shaken[key] {
				delete(e.directSince, key) // proven; stop watching
				continue
			}
			if time.Since(since) > fallbackGrace {
				e.bind.ClearPeerAddr(key)
				delete(e.directSince, key)
				delete(e.applied, key)
			}
		}
		e.mu.Unlock()
	}
}

// PeerPath describes the current packet path for one configured peer.
type PeerPath struct {
	Key           magicsock.NodeKey
	Direct        bool
	LastHandshake time.Time
}

// PeerPaths reports each configured peer's path and last handshake time.
func (e *Engine) PeerPaths() []PeerPath {
	raw, err := e.dev.IpcGet()
	if err != nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	var out []PeerPath
	var idx = -1
	for _, line := range strings.Split(raw, "\n") {
		if v, ok := strings.CutPrefix(line, "public_key="); ok {
			b, err := hex.DecodeString(v)
			if err != nil || len(b) != 32 {
				idx = -1
				continue
			}
			var key magicsock.NodeKey
			copy(key[:], b)
			_, blind := e.applied[key]
			_, proven := e.pmDirect[key]
			out = append(out, PeerPath{Key: key, Direct: blind || proven})
			idx = len(out) - 1
			continue
		}
		if v, ok := strings.CutPrefix(line, "last_handshake_time_sec="); ok && idx >= 0 && v != "0" {
			var sec int64
			fmt.Sscanf(v, "%d", &sec)
			out[idx].LastHandshake = time.Unix(sec, 0)
		}
	}
	return out
}

// handshakedPeers reports which peers have ever completed a handshake.
func (e *Engine) handshakedPeers() map[magicsock.NodeKey]bool {
	out := make(map[magicsock.NodeKey]bool)
	raw, err := e.dev.IpcGet()
	if err != nil {
		return out
	}
	var current magicsock.NodeKey
	var haveCurrent bool
	for _, line := range strings.Split(raw, "\n") {
		if v, ok := strings.CutPrefix(line, "public_key="); ok {
			b, err := hex.DecodeString(v)
			haveCurrent = err == nil && len(b) == 32
			if haveCurrent {
				copy(current[:], b)
			}
			continue
		}
		if v, ok := strings.CutPrefix(line, "last_handshake_time_sec="); ok && haveCurrent && v != "0" {
			out[current] = true
		}
	}
	return out
}

func (e *Engine) LocalPort() uint16 {
	return e.bind.LocalPort()
}

// SetRelaySender wires the coordination session's relay transport into the
// packet path.
func (e *Engine) SetRelaySender(send func(dst uint32, pkt []byte) error) {
	e.bind.SetRelaySender(send)
}

// DeliverRelay feeds a relayed WireGuard packet from the coordination
// session into the packet path.
func (e *Engine) DeliverRelay(src uint32, pkt []byte) {
	e.bind.DeliverRelay(src, pkt)
}

// ApplyNetmap reconfigures the device and path table to match nm. Peers
// absent from nm are removed (replace_peers); each peer's identity is its
// node key, and its freshest known endpoint feeds the Bind's path table.
func (e *Engine) ApplyNetmap(nm netmap.Netmap) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if nm.FilterEnabled {
		e.filter.SetRules(nm.Filter)
	} else {
		e.filter.SetAllowAll()
	}

	var cfg strings.Builder
	cfg.WriteString("replace_peers=true\n")
	inMap := make(map[magicsock.NodeKey]bool, len(nm.Peers))
	for _, p := range nm.Peers {
		key := magicsock.NodeKey(p.NodeKey)
		inMap[key] = true
		fmt.Fprintf(&cfg, "public_key=%s\n", hex.EncodeToString(p.NodeKey[:]))
		fmt.Fprintf(&cfg, "endpoint=%s\n", p.NodeKey)
		fmt.Fprintf(&cfg, "allowed_ip=%s/32\n", p.IP)
		e.bind.SetPeerRelay(key, uint32(p.ID))
		e.keyToID[key] = uint32(p.ID)
		if p.DiscoKey != (netmap.Key{}) {
			// Disco-capable peer: the path manager proves paths; netmap
			// endpoints are just candidates, never blindly trusted.
			e.pm.SetPeer(key, p.DiscoKey, p.Endpoints)
			continue
		}
		if len(p.Endpoints) > 0 && e.applied[key] != p.Endpoints[0] {
			e.bind.SetPeerAddr(key, p.Endpoints[0])
			e.applied[key] = p.Endpoints[0]
			e.directSince[key] = time.Now()
		}
	}
	for key := range e.applied {
		if !inMap[key] {
			e.bind.ClearPeerAddr(key)
			delete(e.applied, key)
			delete(e.directSince, key)
		}
	}
	for key := range e.keyToID {
		if !inMap[key] {
			e.pm.RemovePeer(key)
			e.bind.ClearPeerAddr(key)
			e.bind.RemovePeerHints(key)
			delete(e.pmDirect, key)
			delete(e.keyToID, key)
		}
	}
	return e.dev.IpcSet(cfg.String())
}

func (e *Engine) Close() {
	close(e.done)
	e.dev.Close()
}
