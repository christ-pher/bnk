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

	"golang.zx2c4.com/wireguard/conn"
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

// Bind is a conn.Bind that multiplexes WireGuard, disco, and STUN traffic
// over a single UDP socket.
type Bind struct {
	mu     sync.Mutex
	pc     *net.UDPConn
	port   uint16
	peers  map[NodeKey]netip.AddrPort // identity → current direct address
	byAddr map[netip.AddrPort]NodeKey // reverse map for tagging receives
}

var _ conn.Bind = (*Bind)(nil)

func NewBind() *Bind {
	return &Bind{
		peers:  make(map[NodeKey]netip.AddrPort),
		byAddr: make(map[netip.AddrPort]NodeKey),
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
	return []conn.ReceiveFunc{b.receive}, b.port, nil
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
		key, ok := b.lookupAddr(src)
		if !ok {
			continue
		}
		sizes[0] = n
		eps[0] = &endpoint{key: key}
		return 1, nil
	}
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
	err := b.pc.Close()
	b.pc = nil
	return err
}

func (b *Bind) SetMark(mark uint32) error {
	return nil
}

func (b *Bind) Send(bufs [][]byte, ep conn.Endpoint) error {
	e, ok := ep.(*endpoint)
	if !ok {
		return conn.ErrWrongEndpointType
	}
	b.mu.Lock()
	pc := b.pc
	addr, known := b.peers[e.key]
	b.mu.Unlock()
	if pc == nil {
		return net.ErrClosed
	}
	if !known {
		return fmt.Errorf("%w: %s", ErrNoPath, e.key)
	}
	for _, buf := range bufs {
		if _, err := pc.WriteToUDPAddrPort(buf, addr); err != nil {
			return err
		}
	}
	return nil
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
