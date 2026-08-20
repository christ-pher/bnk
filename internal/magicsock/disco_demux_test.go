package magicsock

import (
	"bytes"
	"net/netip"
	"testing"
	"time"

	"vpnmesh/internal/disco"
)

func discoPkt(payload string) []byte {
	return append([]byte(disco.Magic), payload...)
}

func TestDiscoPacketsGoToHandlerNotWireGuard(t *testing.T) {
	keyA, keyB := newNodeKey(t), newNodeKey(t)
	bindA, bindB := NewBind(), NewBind()
	_, addrA := openBind(t, bindA)
	fnsB, addrB := openBind(t, bindB)
	bindA.SetPeerAddr(keyB, addrB)
	bindB.SetPeerAddr(keyA, addrA)

	got := make(chan []byte, 4)
	srcs := make(chan netip.AddrPort, 4)
	bindB.SetDiscoHandler(func(src netip.AddrPort, pkt []byte) {
		got <- append([]byte(nil), pkt...)
		srcs <- src
	})

	if err := bindA.SendDisco(addrB, discoPkt("probe")); err != nil {
		t.Fatal(err)
	}
	select {
	case pkt := <-got:
		if !bytes.Equal(pkt, discoPkt("probe")) {
			t.Errorf("handler pkt = %q", pkt)
		}
		if src := <-srcs; src.Port() != addrA.Port() {
			t.Errorf("handler src = %v, want port %d", src, addrA.Port())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("disco packet never reached the handler")
	}

	// The same wire still carries WireGuard traffic, and the disco packet
	// must not have surfaced there: the first WG receive is wg-data.
	ep, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("wg-data")}, ep); err != nil {
		t.Fatal(err)
	}
	payload, _ := receiveOne(t, fnsB)
	if !bytes.Equal(payload, []byte("wg-data")) {
		t.Errorf("WG path surfaced %q, want wg-data (disco must not leak into WG)", payload)
	}
}

func TestDiscoFromUnknownSourceStillDelivered(t *testing.T) {
	bindB := NewBind()
	_, addrB := openBind(t, bindB)

	got := make(chan []byte, 1)
	bindB.SetDiscoHandler(func(src netip.AddrPort, pkt []byte) {
		got <- append([]byte(nil), pkt...)
	})

	// A total stranger's disco packet must reach the handler: hole-punch
	// pings arrive from addresses we have never seen.
	stranger := NewBind()
	openBind(t, stranger)
	if err := stranger.SendDisco(addrB, discoPkt("punch")); err != nil {
		t.Fatal(err)
	}
	select {
	case pkt := <-got:
		if !bytes.Equal(pkt, discoPkt("punch")) {
			t.Errorf("pkt = %q", pkt)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("disco from unknown source dropped")
	}
}
