package magicsock

// Relay-path behavior: with no direct address known, Send goes through the
// injected relay sender, and DeliverRelay surfaces packets tagged with the
// peer identity registered for that relay ID. The device-level test proves
// a full WireGuard tunnel over a fake relay hub with zero UDP paths.

import (
	"net/netip"
	"testing"

	"golang.zx2c4.com/wireguard/conn"
)

func TestSendPrefersRelayWhenNoDirectPath(t *testing.T) {
	keyA, keyB := newNodeKey(t), newNodeKey(t)
	bindA := NewBind()
	openBind(t, bindA)

	sent := make(chan []byte, 1)
	bindA.SetRelaySender(func(dst uint32, pkt []byte) error {
		if dst != 2 {
			t.Errorf("relay dst = %d, want 2", dst)
		}
		sent <- append([]byte(nil), pkt...)
		return nil
	})
	bindA.SetPeerRelay(keyB, 2)
	_ = keyA

	ep, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("payload")}, ep); err != nil {
		t.Fatalf("Send via relay: %v", err)
	}
	select {
	case pkt := <-sent:
		if string(pkt) != "payload" {
			t.Errorf("relayed pkt = %q", pkt)
		}
	default:
		t.Fatal("nothing went through the relay sender")
	}
}

func TestDirectPathStillWinsOverRelay(t *testing.T) {
	keyB := newNodeKey(t)
	bindA, bindB := NewBind(), NewBind()
	openBind(t, bindA)
	fnsB, addrB := openBind(t, bindB)

	relayUsed := false
	bindA.SetRelaySender(func(dst uint32, pkt []byte) error {
		relayUsed = true
		return nil
	})
	bindA.SetPeerRelay(keyB, 2)
	bindA.SetPeerAddr(keyB, addrB)
	// bindB must recognize bindA's source addr to surface the packet.
	keyA := newNodeKey(t)
	bindB.SetPeerAddr(keyA, netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), bindA.LocalPort()))

	ep, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("direct")}, ep); err != nil {
		t.Fatal(err)
	}
	payload, _ := receiveOne(t, fnsB)
	if string(payload) != "direct" {
		t.Errorf("payload = %q", payload)
	}
	if relayUsed {
		t.Error("relay used despite a known direct path")
	}
}

func TestDeliverRelaySurfacesIdentityEndpoint(t *testing.T) {
	keyA := newNodeKey(t)
	bindB := NewBind()
	fnsB, _ := openBind(t, bindB)
	bindB.SetPeerRelay(keyA, 1)

	bindB.DeliverRelay(1, []byte("from relay"))

	payload, ep := receiveOne(t, fnsB)
	if string(payload) != "from relay" {
		t.Errorf("payload = %q", payload)
	}
	if ep.DstToString() != keyA.String() {
		t.Errorf("endpoint identity = %q, want %q", ep.DstToString(), keyA)
	}
}

func TestDeliverRelayFromUnknownIDIsDropped(t *testing.T) {
	keyA := newNodeKey(t)
	bindB := NewBind()
	fnsB, _ := openBind(t, bindB)
	bindB.SetPeerRelay(keyA, 1)

	bindB.DeliverRelay(99, []byte("stranger"))
	bindB.DeliverRelay(1, []byte("known"))

	payload, _ := receiveOne(t, fnsB)
	if string(payload) != "known" {
		t.Errorf("first surfaced packet = %q, want the known peer's", payload)
	}
}

func TestTunnelOverRelayOnly(t *testing.T) {
	a := startNode(t, netip.MustParseAddr("100.64.0.1"))
	b := startNode(t, netip.MustParseAddr("100.64.0.2"))

	// Fake hub: no UDP path between the peers at all.
	a.bind.SetRelaySender(func(dst uint32, pkt []byte) error {
		if dst == 2 {
			b.bind.DeliverRelay(1, append([]byte(nil), pkt...))
		}
		return nil
	})
	b.bind.SetRelaySender(func(dst uint32, pkt []byte) error {
		if dst == 1 {
			a.bind.DeliverRelay(2, append([]byte(nil), pkt...))
		}
		return nil
	})
	a.bind.SetPeerRelay(NodeKey(b.pub), 2)
	b.bind.SetPeerRelay(NodeKey(a.pub), 1)
	configurePeer(t, a, b)
	configurePeer(t, b, a)

	echoTCP(t, a, b)
}

var _ conn.Bind = (*Bind)(nil)
