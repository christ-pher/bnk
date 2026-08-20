package magicsock

import (
	"crypto/rand"
	"net"
	"net/netip"
	"testing"
	"time"

	"golang.zx2c4.com/wireguard/conn"
)

func newNodeKey(t *testing.T) NodeKey {
	t.Helper()
	var k NodeKey
	if _, err := rand.Read(k[:]); err != nil {
		t.Fatal(err)
	}
	return k
}

// openBind opens b on a random loopback port and returns its receive
// functions and bound address.
func openBind(t *testing.T, b *Bind) ([]conn.ReceiveFunc, netip.AddrPort) {
	t.Helper()
	fns, port, err := b.Open(0)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(fns) == 0 {
		t.Fatal("Open returned no ReceiveFuncs")
	}
	t.Cleanup(func() { b.Close() })
	return fns, netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), port)
}

// receiveOne blocks until one packet arrives on any of fns and returns its
// payload and endpoint.
func receiveOne(t *testing.T, fns []conn.ReceiveFunc) ([]byte, conn.Endpoint) {
	t.Helper()
	type result struct {
		payload []byte
		ep      conn.Endpoint
	}
	ch := make(chan result, len(fns))
	for _, fn := range fns {
		go func(fn conn.ReceiveFunc) {
			packets := make([][]byte, 1)
			packets[0] = make([]byte, 1500)
			sizes := make([]int, 1)
			eps := make([]conn.Endpoint, 1)
			n, err := fn(packets, sizes, eps)
			if err != nil || n == 0 {
				return
			}
			ch <- result{append([]byte(nil), packets[0][:sizes[0]]...), eps[0]}
		}(fn)
	}
	select {
	case r := <-ch:
		return r.payload, r.ep
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for packet")
		return nil, nil
	}
}

func TestSendDeliversTaggedWithSenderIdentity(t *testing.T) {
	keyA, keyB := newNodeKey(t), newNodeKey(t)
	bindA, bindB := NewBind(), NewBind()
	_, addrA := openBind(t, bindA)
	fnsB, addrB := openBind(t, bindB)

	bindA.SetPeerAddr(keyB, addrB)
	bindB.SetPeerAddr(keyA, addrA)

	epB, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("hello")}, epB); err != nil {
		t.Fatalf("Send: %v", err)
	}

	payload, ep := receiveOne(t, fnsB)
	if string(payload) != "hello" {
		t.Errorf("payload = %q, want %q", payload, "hello")
	}
	if got := ep.DstToString(); got != keyA.String() {
		t.Errorf("received endpoint identity = %q, want sender key %q", got, keyA)
	}
}

func TestSendWithoutKnownPathFails(t *testing.T) {
	bindA := NewBind()
	openBind(t, bindA)

	ep, err := bindA.ParseEndpoint(newNodeKey(t).String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("hello")}, ep); err == nil {
		t.Error("Send to peer with no known path succeeded, want error")
	}
}

func TestPacketsFromUnknownSourcesAreDropped(t *testing.T) {
	keyA, keyB := newNodeKey(t), newNodeKey(t)
	bindA, bindB := NewBind(), NewBind()
	_, addrA := openBind(t, bindA)
	fnsB, addrB := openBind(t, bindB)

	bindA.SetPeerAddr(keyB, addrB)
	bindB.SetPeerAddr(keyA, addrA)

	// A stranger's datagram must not surface from the ReceiveFunc.
	stranger, err := net.Dial("udp", addrB.String())
	if err != nil {
		t.Fatal(err)
	}
	defer stranger.Close()
	if _, err := stranger.Write([]byte("intruder")); err != nil {
		t.Fatal(err)
	}

	// A legitimate packet sent afterwards must be the first thing received.
	epB, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("legit")}, epB); err != nil {
		t.Fatalf("Send: %v", err)
	}

	payload, ep := receiveOne(t, fnsB)
	if string(payload) != "legit" {
		t.Errorf("first received payload = %q, want %q (unknown source must be dropped)", payload, "legit")
	}
	if got := ep.DstToString(); got != keyA.String() {
		t.Errorf("received endpoint identity = %q, want %q", got, keyA)
	}
}
