package magicsock

import (
	"net/netip"
	"testing"
)

func TestClearPeerAddrFallsBackToRelay(t *testing.T) {
	keyB := newNodeKey(t)
	bindA := NewBind()
	openBind(t, bindA)

	relayed := make(chan []byte, 4)
	bindA.SetRelaySender(func(dst uint32, pkt []byte) error {
		relayed <- append([]byte(nil), pkt...)
		return nil
	})
	bindA.SetPeerRelay(keyB, 2)
	bindA.SetPeerAddr(keyB, netip.MustParseAddrPort("127.0.0.1:9")) // discard

	ep, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("one")}, ep); err != nil {
		t.Fatal(err)
	}
	select {
	case pkt := <-relayed:
		t.Fatalf("packet %q used relay while a direct addr was set", pkt)
	default:
	}

	bindA.ClearPeerAddr(keyB)
	if err := bindA.Send([][]byte{[]byte("two")}, ep); err != nil {
		t.Fatal(err)
	}
	select {
	case pkt := <-relayed:
		if string(pkt) != "two" {
			t.Errorf("relayed pkt = %q, want %q", pkt, "two")
		}
	default:
		t.Fatal("Send after ClearPeerAddr did not use the relay")
	}
}
