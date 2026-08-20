package magicsock

// Regression for the asymmetric-promotion bug: A sends to B via one of
// B's addresses while B sends from another (both valid candidates). The
// receive path must attribute inbound WireGuard packets by ANY known
// candidate address of the peer, not only the address we send to.

import (
	"net/netip"
	"testing"
)

func TestInboundAttributedByAddrHintNotJustSendAddr(t *testing.T) {
	keyB := newNodeKey(t)
	bindA, bindB := NewBind(), NewBind()
	fnsA, addrA := openBind(t, bindA)
	_, addrB := openBind(t, bindB)

	// A believes B lives at a different address than B actually sends
	// from (the mismatched-family scenario).
	bindA.SetPeerAddr(keyB, netip.MustParseAddrPort("192.0.2.99:1234"))
	// But B's real source address is a known candidate, registered as a hint.
	bindA.AddAddrHint(keyB, addrB)

	// B sends A a WireGuard packet from its real socket.
	keyA := newNodeKey(t)
	bindB.SetPeerAddr(keyA, addrA)
	ep, err := bindB.ParseEndpoint(keyA.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindB.Send([][]byte{[]byte("wg-from-real-addr")}, ep); err != nil {
		t.Fatal(err)
	}

	payload, gotEp := receiveOne(t, fnsA)
	if string(payload) != "wg-from-real-addr" {
		t.Fatalf("payload = %q (packet from hinted addr was dropped)", payload)
	}
	if gotEp.DstToString() != keyB.String() {
		t.Errorf("attributed to %q, want %q", gotEp.DstToString(), keyB)
	}
}

func TestRemovePeerHintsForgetsAttribution(t *testing.T) {
	keyB := newNodeKey(t)
	bindA := NewBind()
	openBind(t, bindA)

	hint := netip.MustParseAddrPort("192.0.2.50:5555")
	bindA.AddAddrHint(keyB, hint)
	if got, ok := bindA.lookupAddr(hint); !ok || got != keyB {
		t.Fatal("hint not registered")
	}
	bindA.RemovePeerHints(keyB)
	if _, ok := bindA.lookupAddr(hint); ok {
		t.Error("hint survived RemovePeerHints")
	}
}
