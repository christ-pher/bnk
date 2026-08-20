// Package magicsock implements a wireguard-go conn.Bind whose endpoints
// identify peers by node key rather than ip:port, so the socket layer can
// switch a peer between direct and relayed paths without WireGuard noticing.
package magicsock

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"sync"
	"sync/atomic"

	"golang.zx2c4.com/wireguard/conn"

	"vpnmesh/internal/disco"
)

// NodeKey is a peer's WireGuard public key, the stable identity every path
// (direct or relayed) maps back to.
type NodeKey [32]byte

func ParseNodeKey(s string) (NodeKey, error) {
	var k NodeKey
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return k, fmt.Errorf("node key %q: %w", s, err)
	}
	if len(raw) != len(k) {
		return k, fmt.Errorf("node key %q: got %d bytes, want %d", s, len(raw), len(k))
	}
	copy(k[:], raw)
	return k, nil
}

func (k NodeKey) String() string {
	return base64.StdEncoding.EncodeToString(k[:])
}

var ErrNoPath = errors.New("magicsock: no known path to peer")

type relayPacket struct {
	key NodeKey
	pkt []byte
}

// Bind is a conn.Bind that multiplexes WireGuard, disco, and STUN traffic
// over a single UDP socket.
type Bind struct {
	mu        sync.Mutex
	pc        *net.UDPConn
	port      uint16
	peers     map[NodeKey]netip.AddrPort // identity → current direct address
	byAddr    map[netip.AddrPort]NodeKey // reverse map for tagging receives
	relaySend func(dst uint32, pkt []byte) error
	relayIDs  map[NodeKey]uint32
	byRelayID map[uint32]NodeKey
	relayCh   chan relayPacket
	closeCh   chan struct{}
	onDisco   func(src netip.AddrPort, pkt []byte)
	onSTUN    func(pkt []byte)
	relayTx   atomic.Uint64
	relayRx   atomic.Uint64
}

var _ conn.Bind = (*Bind)(nil)

func NewBind() *Bind {
	return &Bind{
		peers:     make(map[NodeKey]netip.AddrPort),
		byAddr:    make(map[netip.AddrPort]NodeKey),
		relayIDs:  make(map[NodeKey]uint32),
		byRelayID: make(map[uint32]NodeKey),
		relayCh:   make(chan relayPacket, 64),
	}
}

func (b *Bind) Open(port uint16) ([]conn.ReceiveFunc, uint16, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.pc != nil {
		return nil, 0, conn.ErrBindAlreadyOpen
	}
	pc, err := net.ListenUDP("udp", &net.UDPAddr{Port: int(port)})
	if err != nil {
		return nil, 0, err
	}
	b.pc = pc
	b.port = uint16(pc.LocalAddr().(*net.UDPAddr).Port)
	b.closeCh = make(chan struct{})
	return []conn.ReceiveFunc{b.receive, b.receiveRelay}, b.port, nil
}

// receiveRelay surfaces packets handed in by DeliverRelay as the Bind's
// second receive path (the same way Tailscale plumbs DERP receives).
func (b *Bind) receiveRelay(packets [][]byte, sizes []int, eps []conn.Endpoint) (int, error) {
	b.mu.Lock()
	closeCh := b.closeCh
	b.mu.Unlock()
	if closeCh == nil {
		return 0, net.ErrClosed
	}
	select {
	case <-closeCh:
		return 0, net.ErrClosed
	case rp := <-b.relayCh:
		n := copy(packets[0], rp.pkt)
		sizes[0] = n
		eps[0] = &endpoint{key: rp.key}
		return 1, nil
	}
}

// receive reads datagrams until one comes from a known peer, tagging it with
// that peer's identity endpoint. Unknown sources are dropped: only the path
// table (fed by the control plane and, later, proven disco paths) grants a
// source address a peer identity.
func (b *Bind) receive(packets [][]byte, sizes []int, eps []conn.Endpoint) (int, error) {
	b.mu.Lock()
	pc := b.pc
	b.mu.Unlock()
	if pc == nil {
		return 0, net.ErrClosed
	}
	for {
		n, src, err := pc.ReadFromUDPAddrPort(packets[0])
		if err != nil {
			return 0, err
		}
		// Disco traffic peels off before any peer check: hole-punch probes
		// arrive from source addresses no table knows yet.
		if disco.IsDisco(packets[0][:n]) {
			b.mu.Lock()
			h := b.onDisco
			b.mu.Unlock()
			if h != nil {
				h(normalize(src), append([]byte(nil), packets[0][:n]...))
			}
			continue
		}
		if isSTUN(packets[0][:n]) {
			b.mu.Lock()
			h := b.onSTUN
			b.mu.Unlock()
			if h != nil {
				h(append([]byte(nil), packets[0][:n]...))
			}
			continue
		}
		key, ok := b.lookupAddr(src)
		if !ok {
			continue
		}
		sizes[0] = n
		eps[0] = &endpoint{key: key}
		return 1, nil
	}
}

// isSTUN classifies a datagram by the RFC 5389 shape: top two bits zero
// and the magic cookie at bytes 4-8. WireGuard packets start 0x01-0x04
// followed by three zero bytes, so they can never look like this.
func isSTUN(pkt []byte) bool {
	return len(pkt) >= 20 && pkt[0]>>6 == 0 &&
		pkt[4] == 0x21 && pkt[5] == 0x12 && pkt[6] == 0xA4 && pkt[7] == 0x42
}

// SetSTUNHandler registers the callback for inbound STUN responses. The
// callback runs on the receive goroutine and must not block.
func (b *Bind) SetSTUNHandler(h func(pkt []byte)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.onSTUN = h
}

func (b *Bind) lookupAddr(src netip.AddrPort) (NodeKey, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	key, ok := b.byAddr[normalize(src)]
	return key, ok
}

// normalize maps 4-in-6 addresses (as returned by a dual-stack socket) to
// plain IPv4 so table entries match regardless of representation.
func normalize(ap netip.AddrPort) netip.AddrPort {
	return netip.AddrPortFrom(ap.Addr().Unmap(), ap.Port())
}

func (b *Bind) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.pc == nil {
		return nil
	}
	close(b.closeCh)
	b.closeCh = nil
	err := b.pc.Close()
	b.pc = nil
	return err
}

func (b *Bind) SetMark(mark uint32) error {
	return nil
}

// Send routes to the peer's direct address when one is known, else falls
// back to the relay. WireGuard never sees which path was taken.
func (b *Bind) Send(bufs [][]byte, ep conn.Endpoint) error {
	e, ok := ep.(*endpoint)
	if !ok {
		return conn.ErrWrongEndpointType
	}
	b.mu.Lock()
	pc := b.pc
	addr, direct := b.peers[e.key]
	relayID, hasRelay := b.relayIDs[e.key]
	relaySend := b.relaySend
	b.mu.Unlock()
	if pc == nil {
		return net.ErrClosed
	}
	switch {
	case direct:
		for _, buf := range bufs {
			if _, err := pc.WriteToUDPAddrPort(buf, addr); err != nil {
				return err
			}
		}
		return nil
	case hasRelay && relaySend != nil:
		for _, buf := range bufs {
			if err := relaySend(relayID, buf); err != nil {
				return err
			}
			b.relayTx.Add(1)
		}
		return nil
	default:
		return fmt.Errorf("%w: %s", ErrNoPath, e.key)
	}
}

func (b *Bind) BatchSize() int {
	return 1
}

// LocalPort reports the UDP port the Bind is listening on, or 0 if closed.
func (b *Bind) LocalPort() uint16 {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.pc == nil {
		return 0
	}
	return b.port
}

// SetDiscoHandler registers the callback for inbound disco packets. The
// callback runs on the receive goroutine and must not block.
func (b *Bind) SetDiscoHandler(h func(src netip.AddrPort, pkt []byte)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.onDisco = h
}

// SendRaw transmits a raw datagram (disco probe, STUN query) to addr,
// independent of any peer path state — this is how candidate paths are
// probed before they exist.
func (b *Bind) SendRaw(addr netip.AddrPort, pkt []byte) error {
	b.mu.Lock()
	pc := b.pc
	b.mu.Unlock()
	if pc == nil {
		return net.ErrClosed
	}
	_, err := pc.WriteToUDPAddrPort(pkt, addr)
	return err
}

// SetRelaySender injects the transport used to reach peers with no direct
// path (frames via the coordination session, in production).
func (b *Bind) SetRelaySender(send func(dst uint32, pkt []byte) error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.relaySend = send
}

// SetPeerRelay registers a peer's relay ID, enabling the relay fallback
// path for it and mapping inbound relay packets back to its identity.
func (b *Bind) SetPeerRelay(key NodeKey, id uint32) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if old, ok := b.relayIDs[key]; ok {
		delete(b.byRelayID, old)
	}
	b.relayIDs[key] = id
	b.byRelayID[id] = key
}

// DeliverRelay hands the Bind one WireGuard packet that arrived via the
// relay from the peer registered under src. Packets from unregistered IDs
// are dropped; if the queue is full the packet is dropped (UDP semantics).
func (b *Bind) DeliverRelay(src uint32, pkt []byte) {
	b.mu.Lock()
	key, ok := b.byRelayID[src]
	b.mu.Unlock()
	if !ok {
		return
	}
	select {
	case b.relayCh <- relayPacket{key: key, pkt: pkt}:
		b.relayRx.Add(1)
	default:
	}
}

// RelayStats reports how many WireGuard packets have gone out via and
// come in from the relay — the first question when debugging "why is
// there no tunnel".
func (b *Bind) RelayStats() (tx, rx uint64) {
	return b.relayTx.Load(), b.relayRx.Load()
}

// SetPeerAddr sets the current direct address for a peer. Phase 0: a static
// passthrough table; later phases hand this to the path manager.
func (b *Bind) SetPeerAddr(key NodeKey, addr netip.AddrPort) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if old, ok := b.peers[key]; ok {
		delete(b.byAddr, normalize(old))
	}
	b.peers[key] = addr
	b.byAddr[normalize(addr)] = key
}

// ClearPeerAddr removes a peer's direct address so sends fall back to the
// relay (used when a direct path never proves itself).
func (b *Bind) ClearPeerAddr(key NodeKey) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if old, ok := b.peers[key]; ok {
		delete(b.byAddr, normalize(old))
		delete(b.peers, key)
	}
}

// ParseEndpoint accepts a base64 node key, not an ip:port: WireGuard
// addresses peers by identity and the Bind owns the identity→address mapping.
func (b *Bind) ParseEndpoint(s string) (conn.Endpoint, error) {
	key, err := ParseNodeKey(s)
	if err != nil {
		return nil, err
	}
	return &endpoint{key: key}, nil
}

type endpoint struct {
	key NodeKey
}

func (e *endpoint) ClearSrc()           {}
func (e *endpoint) SrcToString() string { return "" }
func (e *endpoint) DstToString() string { return e.key.String() }
func (e *endpoint) DstToBytes() []byte  { return e.key[:] }
func (e *endpoint) DstIP() netip.Addr   { return netip.Addr{} }
func (e *endpoint) SrcIP() netip.Addr   { return netip.Addr{} }
