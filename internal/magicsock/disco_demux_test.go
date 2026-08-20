package magicsock

import (
	"bytes"
	"net/netip"
	"testing"
	"time"

	"golang.zx2c4.com/wireguard/conn"

	"github.com/christ-pher/bnk/internal/disco"
)

func discoPkt(payload string) []byte {
	return append([]byte(disco.Magic), payload...)
}

// pump drives a Bind's ReceiveFuncs the way wireguard-go does, forwarding
// surfaced WG packets to the returned channel. Without a pump the demux
// never runs.
func pump(t *testing.T, fns []conn.ReceiveFunc) <-chan []byte {
	t.Helper()
	out := make(chan []byte, 16)
	for _, fn := range fns {
		go func(fn conn.ReceiveFunc) {
			for {
				packets := [][]byte{make([]byte, 1500)}
				sizes := make([]int, 1)
				eps := make([]conn.Endpoint, 1)
				n, err := fn(packets, sizes, eps)
				if err != nil {
					return
				}
				if n > 0 {
					out <- append([]byte(nil), packets[0][:sizes[0]]...)
				}
			}
		}(fn)
	}
	return out
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
	wgGot := pump(t, fnsB)

	if err := bindA.SendRaw(addrB, discoPkt("probe")); err != nil {
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
	select {
	case payload := <-wgGot:
		if !bytes.Equal(payload, []byte("wg-data")) {
			t.Errorf("WG path surfaced %q, want wg-data (disco must not leak into WG)", payload)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("WG packet never surfaced")
	}
}

func TestDiscoFromUnknownSourceStillDelivered(t *testing.T) {
	bindB := NewBind()
	fnsB, addrB := openBind(t, bindB)

	got := make(chan []byte, 1)
	bindB.SetDiscoHandler(func(src netip.AddrPort, pkt []byte) {
		got <- append([]byte(nil), pkt...)
	})
	pump(t, fnsB)

	// A total stranger's disco packet must reach the handler: hole-punch
	// pings arrive from addresses we have never seen.
	stranger := NewBind()
	openBind(t, stranger)
	if err := stranger.SendRaw(addrB, discoPkt("punch")); err != nil {
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
