package filter

import (
	"encoding/binary"
	"net/netip"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/acl"
)

// ipv4 builds a minimal IPv4 packet with the given payload attached.
func ipv4(proto byte, src, dst string, payload []byte) []byte {
	pkt := make([]byte, 20+len(payload))
	pkt[0] = 0x45 // v4, 20-byte header
	binary.BigEndian.PutUint16(pkt[2:], uint16(len(pkt)))
	pkt[8] = 64
	pkt[9] = proto
	copy(pkt[12:16], netip.MustParseAddr(src).AsSlice())
	copy(pkt[16:20], netip.MustParseAddr(dst).AsSlice())
	copy(pkt[20:], payload)
	return pkt
}

func tcp(src, dst string, sport, dport uint16, syn bool) []byte {
	hdr := make([]byte, 20)
	binary.BigEndian.PutUint16(hdr[0:], sport)
	binary.BigEndian.PutUint16(hdr[2:], dport)
	hdr[12] = 5 << 4
	if syn {
		hdr[13] = 0x02
	} else {
		hdr[13] = 0x10 // ACK
	}
	return ipv4(6, src, dst, hdr)
}

func udp(src, dst string, sport, dport uint16) []byte {
	hdr := make([]byte, 8)
	binary.BigEndian.PutUint16(hdr[0:], sport)
	binary.BigEndian.PutUint16(hdr[2:], dport)
	binary.BigEndian.PutUint16(hdr[4:], 8)
	return ipv4(17, src, dst, hdr)
}

func icmpEcho(src, dst string, reply bool) []byte {
	hdr := make([]byte, 8)
	if reply {
		hdr[0] = 0
	} else {
		hdr[0] = 8
	}
	return ipv4(1, src, dst, hdr)
}

var (
	peer  = "100.64.0.9"
	other = "100.64.0.7"
	self  = "100.64.0.1"
	t0    = time.Unix(1700000000, 0)
)

func sshOnlyFilter() *Filter {
	f := New()
	f.SetRules([]acl.CompiledRule{{
		Srcs:  []netip.Prefix{netip.MustParsePrefix(peer + "/32")},
		Proto: "tcp",
		Ports: []acl.PortRange{{First: 22, Last: 22}},
	}})
	return f
}

func TestAllowAllModePassesEverything(t *testing.T) {
	f := New()
	if !f.CheckInbound(tcp(peer, self, 5555, 23, true), t0) {
		t.Error("allow-all dropped a packet")
	}
}

func TestRuleAllowsMatchingInbound(t *testing.T) {
	f := sshOnlyFilter()
	if !f.CheckInbound(tcp(peer, self, 5555, 22, true), t0) {
		t.Error("rule-allowed tcp/22 dropped")
	}
}

func TestDefaultDenyDropsNonMatching(t *testing.T) {
	f := sshOnlyFilter()
	if f.CheckInbound(tcp(peer, self, 5555, 23, true), t0) {
		t.Error("tcp/23 passed despite ssh-only policy")
	}
	if f.CheckInbound(tcp(other, self, 5555, 22, true), t0) {
		t.Error("tcp/22 from disallowed source passed")
	}
	if f.CheckInbound(udp(peer, self, 53, 22), t0) {
		t.Error("udp passed a tcp-only rule")
	}
}

func TestReturnTrafficAllowedViaFlowTable(t *testing.T) {
	f := sshOnlyFilter()
	// We initiate outbound to peer:9999; replies must come back in.
	f.NoteOutbound(udp(self, peer, 40000, 9999), t0)
	if !f.CheckInbound(udp(peer, self, 9999, 40000), t0.Add(2*time.Second)) {
		t.Error("UDP reply to our own outbound flow was dropped")
	}
	// Same reply after the UDP flow TTL must be dropped.
	if f.CheckInbound(udp(peer, self, 9999, 40000), t0.Add(5*time.Minute)) {
		t.Error("UDP reply passed long after the flow expired")
	}
	// A different remote port is a different flow.
	if f.CheckInbound(udp(peer, self, 9998, 40000), t0.Add(time.Second)) {
		t.Error("UDP from a different source port matched the flow")
	}
}

func TestTCPReturnFlowAllowed(t *testing.T) {
	f := sshOnlyFilter()
	f.NoteOutbound(tcp(self, peer, 40000, 8080, true), t0)
	if !f.CheckInbound(tcp(peer, self, 8080, 40000, false), t0.Add(time.Second)) {
		t.Error("TCP reply to our outbound connection was dropped")
	}
	if f.CheckInbound(tcp(peer, self, 8081, 40000, false), t0.Add(time.Second)) {
		t.Error("TCP from wrong port matched the flow")
	}
}

func TestICMPEchoReplyViaFlow(t *testing.T) {
	f := sshOnlyFilter()
	f.NoteOutbound(icmpEcho(self, peer, false), t0)
	if !f.CheckInbound(icmpEcho(peer, self, true), t0.Add(time.Second)) {
		t.Error("ICMP echo reply to our ping was dropped")
	}
	if f.CheckInbound(icmpEcho(other, self, true), t0.Add(time.Second)) {
		t.Error("ICMP reply from a host we never pinged passed")
	}
}

func TestICMPRuleAllowsInboundPing(t *testing.T) {
	f := New()
	f.SetRules([]acl.CompiledRule{{
		Srcs:  []netip.Prefix{netip.MustParsePrefix(peer + "/32")},
		Proto: "icmp",
	}})
	if !f.CheckInbound(icmpEcho(peer, self, false), t0) {
		t.Error("icmp rule did not allow inbound ping")
	}
}

func TestMalformedAndNonIPv4Dropped(t *testing.T) {
	f := sshOnlyFilter()
	if f.CheckInbound([]byte{0x60, 0x00, 0x00}, t0) {
		t.Error("IPv6/short packet passed while enforcing")
	}
	if f.CheckInbound(nil, t0) {
		t.Error("nil packet passed")
	}
	truncated := tcp(peer, self, 1, 22, true)[:22]
	if f.CheckInbound(truncated, t0) {
		t.Error("truncated transport header passed")
	}
}

func TestSetAllowAllDisablesEnforcement(t *testing.T) {
	f := sshOnlyFilter()
	f.SetAllowAll()
	if !f.CheckInbound(tcp(other, self, 5555, 23, true), t0) {
		t.Error("allow-all after rules still dropped")
	}
}
