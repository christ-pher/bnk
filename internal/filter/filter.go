// Package filter is the client-side ACL enforcement point: a userspace
// packet filter sitting between the WireGuard device and the TUN, identical
// on every platform. Default-deny inbound when rules are set, with a flow
// table admitting return traffic for connections this node initiated.
package filter

import (
	"encoding/binary"
	"net/netip"
	"sync"
	"time"

	"golang.zx2c4.com/wireguard/tun"

	"github.com/christ-pher/bnk/internal/acl"
)

const (
	udpFlowTTL  = 30 * time.Second
	tcpFlowTTL  = 2 * time.Minute
	icmpFlowTTL = 30 * time.Second
)

// flowKey identifies a connection from the local node's perspective:
// (remote addr, remote port, local port, proto). ICMP uses zero ports.
type flowKey struct {
	remote     netip.Addr
	remotePort uint16
	localPort  uint16
	proto      byte
}

type Filter struct {
	mu      sync.Mutex
	enforce bool
	rules   []acl.CompiledRule
	flows   map[flowKey]time.Time // expiry
}

// New returns a filter in allow-all mode (no policy pushed yet).
func New() *Filter {
	return &Filter{flows: make(map[flowKey]time.Time)}
}

// SetRules enables enforcement with the given inbound allowances.
func (f *Filter) SetRules(rules []acl.CompiledRule) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.enforce = true
	f.rules = rules
}

// SetAllowAll returns the filter to pass-everything mode.
func (f *Filter) SetAllowAll() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.enforce = false
	f.rules = nil
}

// parsed is the subset of header fields the filter matches on.
type parsed struct {
	proto     byte // 1 icmp, 6 tcp, 17 udp
	src, dst  netip.Addr
	srcPort   uint16
	dstPort   uint16
	protoName string
	icmpType  byte
}

func parse(pkt []byte) (parsed, bool) {
	if len(pkt) < 20 || pkt[0]>>4 != 4 {
		return parsed{}, false
	}
	ihl := int(pkt[0]&0x0F) * 4
	if ihl < 20 || len(pkt) < ihl {
		return parsed{}, false
	}
	p := parsed{proto: pkt[9]}
	src, _ := netip.AddrFromSlice(pkt[12:16])
	dst, _ := netip.AddrFromSlice(pkt[16:20])
	p.src, p.dst = src, dst
	body := pkt[ihl:]
	switch p.proto {
	case 6, 17:
		if len(body) < 4 {
			return parsed{}, false
		}
		p.srcPort = binary.BigEndian.Uint16(body[0:2])
		p.dstPort = binary.BigEndian.Uint16(body[2:4])
		if p.proto == 6 {
			p.protoName = "tcp"
		} else {
			p.protoName = "udp"
		}
	case 1:
		if len(body) < 1 {
			return parsed{}, false
		}
		p.icmpType = body[0]
		p.protoName = "icmp"
	default:
		return parsed{}, false
	}
	return p, true
}

func flowTTL(proto byte) time.Duration {
	switch proto {
	case 6:
		return tcpFlowTTL
	case 1:
		return icmpFlowTTL
	default:
		return udpFlowTTL
	}
}

// CheckInbound decides whether a decrypted packet from a peer may reach
// the local host.
func (f *Filter) CheckInbound(pkt []byte, now time.Time) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.enforce {
		return true
	}
	p, ok := parse(pkt)
	if !ok {
		return false
	}
	for _, r := range f.rules {
		if r.Match(p.src, p.protoName, p.dstPort) {
			return true
		}
	}
	// Return traffic: inbound (src,srcPort→dstPort) mirrors a recorded
	// outbound flow (remote=src, remotePort=srcPort, localPort=dstPort).
	key := flowKey{remote: p.src, remotePort: p.srcPort, localPort: p.dstPort, proto: p.proto}
	if exp, ok := f.flows[key]; ok {
		if now.Before(exp) {
			f.flows[key] = now.Add(flowTTL(p.proto)) // keep-alive on traffic
			return true
		}
		delete(f.flows, key)
	}
	return false
}

// WrapTUN interposes the filter between the WireGuard device and the OS:
// inbound Writes are checked, outbound Reads feed the flow table.
func WrapTUN(inner tun.Device, f *Filter) tun.Device {
	return &filteredTUN{Device: inner, f: f}
}

type filteredTUN struct {
	tun.Device
	f *Filter
}

func (t *filteredTUN) Read(bufs [][]byte, sizes []int, offset int) (int, error) {
	n, err := t.Device.Read(bufs, sizes, offset)
	now := time.Now()
	for i := 0; i < n; i++ {
		t.f.NoteOutbound(bufs[i][offset:offset+sizes[i]], now)
	}
	return n, err
}

// Write drops denied packets in place but still reports them written:
// WireGuard treats short writes as errors, and a filtered drop is not one.
func (t *filteredTUN) Write(bufs [][]byte, offset int) (int, error) {
	now := time.Now()
	kept := bufs[:0]
	for _, b := range bufs {
		if t.f.CheckInbound(b[offset:], now) {
			kept = append(kept, b)
		}
	}
	if len(kept) == 0 {
		return len(bufs), nil
	}
	if _, err := t.Device.Write(kept, offset); err != nil {
		return 0, err
	}
	return len(bufs), nil
}

// NoteOutbound records locally-originated traffic so replies pass.
func (f *Filter) NoteOutbound(pkt []byte, now time.Time) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.enforce {
		return
	}
	p, ok := parse(pkt)
	if !ok {
		return
	}
	key := flowKey{remote: p.dst, remotePort: p.dstPort, localPort: p.srcPort, proto: p.proto}
	f.flows[key] = now.Add(flowTTL(p.proto))
	// Opportunistic expiry sweep, amortized and bounded.
	if len(f.flows) > 4096 {
		for k, exp := range f.flows {
			if now.After(exp) {
				delete(f.flows, k)
			}
		}
	}
}
