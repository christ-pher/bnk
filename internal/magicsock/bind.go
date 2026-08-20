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
	"runtime"
	"sync"
	"sync/atomic"

	"golang.org/x/net/ipv6"
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
	br        *ipv6.PacketConn // batch reader/writer over pc (recvmmsg/sendmmsg)
	rmsgs     []ipv6.Message   // receive staging; owned by the single receive goroutine
	smsgPool  sync.Pool        // *[]ipv6.Message for concurrent senders
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
	b.br = ipv6.NewPacketConn(pc)
	b.smsgPool.New = func() any {
		msgs := make([]ipv6.Message, conn.IdealBatchSize)
		for i := range msgs {
			msgs[i].Buffers = make([][]byte, 1)
		}
		return &msgs
	}
	b.port = uint16(pc.LocalAddr().(*net.UDPAddr).Port)
	b.closeCh = make(chan struct{})
	return []conn.ReceiveFunc{b.receive, b.receiveRelay}, b.port, nil
}

// receiveRelay surfaces packets handed in by DeliverRelay as the Bind's
// second receive path (the same way Tailscale plumbs DERP receives). It
// drains whatever has queued up to the batch size before returning.
func (b *Bind) receiveRelay(packets [][]byte, sizes []int, eps []conn.Endpoint) (int, error) {
	b.mu.Lock()
	closeCh := b.closeCh
	b.mu.Unlock()
	if closeCh == nil {
		return 0, net.ErrClosed
	}
	fill := func(i int, rp relayPacket) {
		sizes[i] = copy(packets[i], rp.pkt)
		eps[i] = &endpoint{key: rp.key}
	}
	select {
	case <-closeCh:
		return 0, net.ErrClosed
	case rp := <-b.relayCh:
		fill(0, rp)
		n := 1
		for n < len(packets) {
			select {
			case rp := <-b.relayCh:
				fill(n, rp)
				n++
			default:
				return n, nil
			}
		}
		return n, nil
	}
}

// batchIO reports whether recvmmsg/sendmmsg batching is safe to use.
// x/net only implements the batch syscalls on Linux; elsewhere the
// classic one-datagram calls are the correct path.
func batchIO() bool {
	return runtime.GOOS == "linux"
}

// receive reads datagrams in batches (recvmmsg) until at least one comes
// from a known peer, tagging each with that peer's identity endpoint.
// Unknown sources are dropped: only the path table (fed by the control
// plane and proven disco paths) grants a source address a peer identity.
// Called serially from a single wireguard-go goroutine.
func (b *Bind) receive(packets [][]byte, sizes []int, eps []conn.Endpoint) (int, error) {
	b.mu.Lock()
	pc, br := b.pc, b.br
	b.mu.Unlock()
	if pc == nil {
		return 0, net.ErrClosed
	}
	if !batchIO() {
		return b.receiveOne(pc, packets, sizes, eps)
	}
	if len(b.rmsgs) < len(packets) {
		b.rmsgs = make([]ipv6.Message, len(packets))
	}
	for {
		msgs := b.rmsgs[:len(packets)]
		for i := range msgs {
			// Read straight into the caller's buffers; kept packets from a
			// mixed batch are compacted below.
			msgs[i].Buffers = [][]byte{packets[i]}
			msgs[i].Addr = nil
			msgs[i].N = 0
		}
		n, err := br.ReadBatch(msgs, 0)
		if err != nil {
			return 0, err
		}
		out := 0
		for i := 0; i < n; i++ {
			pkt := packets[i][:msgs[i].N]
			ua, ok := msgs[i].Addr.(*net.UDPAddr)
			if !ok {
				continue
			}
			key, keep := b.classify(pkt, ua.AddrPort())
			if !keep {
				continue
			}
			if out != i {
				copy(packets[out], pkt)
			}
			sizes[out] = len(pkt)
			eps[out] = &endpoint{key: key}
			out++
		}
		if out > 0 {
			return out, nil
		}
	}
}

// classify demuxes one datagram: disco and STUN are dispatched to their
// handlers (keep=false), known-peer WireGuard traffic returns its identity
// (keep=true), and unknown sources are dropped. Disco peels off before any
// peer check: hole-punch probes arrive from addresses no table knows yet.
func (b *Bind) classify(pkt []byte, src netip.AddrPort) (key NodeKey, keep bool) {
	if disco.IsDisco(pkt) {
		b.mu.Lock()
		h := b.onDisco
		b.mu.Unlock()
		if h != nil {
			h(normalize(src), append([]byte(nil), pkt...))
		}
		return NodeKey{}, false
	}
	if isSTUN(pkt) {
		b.mu.Lock()
		h := b.onSTUN
		b.mu.Unlock()
		if h != nil {
			h(append([]byte(nil), pkt...))
		}
		return NodeKey{}, false
	}
	key, ok := b.lookupAddr(src)
	return key, ok
}

// receiveOne is the non-Linux receive path: one datagram per syscall.
func (b *Bind) receiveOne(pc *net.UDPConn, packets [][]byte, sizes []int, eps []conn.Endpoint) (int, error) {
	for {
		n, src, err := pc.ReadFromUDPAddrPort(packets[0])
		if err != nil {
			return 0, err
		}
		key, keep := b.classify(packets[0][:n], src)
		if !keep {
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
	b.br = nil
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
		return b.sendBatch(bufs, addr)
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

// sendBatch writes bufs to addr with as few syscalls as possible
// (sendmmsg). Send is called concurrently by per-peer goroutines, so the
// message headers come from a pool rather than the Bind.
func (b *Bind) sendBatch(bufs [][]byte, addr netip.AddrPort) error {
	b.mu.Lock()
	br := b.br
	b.mu.Unlock()
	if br == nil {
		return net.ErrClosed
	}
	if !batchIO() {
		b.mu.Lock()
		pc := b.pc
		b.mu.Unlock()
		if pc == nil {
			return net.ErrClosed
		}
		for _, buf := range bufs {
			if _, err := pc.WriteToUDPAddrPort(buf, addr); err != nil {
				return err
			}
		}
		return nil
	}
	ua := net.UDPAddrFromAddrPort(addr)
	msgsp := b.smsgPool.Get().(*[]ipv6.Message)
	defer b.smsgPool.Put(msgsp)
	msgs := *msgsp
	for len(bufs) > 0 {
		chunk := min(len(bufs), len(msgs))
		for i := 0; i < chunk; i++ {
			msgs[i].Buffers[0] = bufs[i]
			msgs[i].Addr = ua
		}
		for sent := 0; sent < chunk; {
			n, err := br.WriteBatch(msgs[sent:chunk], 0)
			if err != nil {
				return err
			}
			sent += n
		}
		bufs = bufs[chunk:]
	}
	return nil
}

func (b *Bind) BatchSize() int {
	if !batchIO() {
		return 1
	}
	return conn.IdealBatchSize
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

// AddAddrHint registers an additional source address that should be
// attributed to peer key on receive. The peer may send from any of its
// candidate addresses, not just the one we send to.
func (b *Bind) AddAddrHint(key NodeKey, addr netip.AddrPort) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.byAddr[normalize(addr)] = key
}

// RemovePeerHints forgets every source-address attribution for key.
func (b *Bind) RemovePeerHints(key NodeKey) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for addr, k := range b.byAddr {
		if k == key {
			delete(b.byAddr, addr)
		}
	}
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

// SetPeerAddr sets the peer's current send address. Old reverse-map
// entries are kept: the peer may legitimately still send from any of its
// candidate addresses (see AddAddrHint), and forgetting them silently
// drops its WireGuard traffic.
func (b *Bind) SetPeerAddr(key NodeKey, addr netip.AddrPort) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.peers[key] = addr
	b.byAddr[normalize(addr)] = key
}

// ClearPeerAddr removes a peer's direct send address so sends fall back
// to the relay. Receive attribution (byAddr) is kept: stray packets from
// the dead path are harmless, dropped ones are not.
func (b *Bind) ClearPeerAddr(key NodeKey) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.peers, key)
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
