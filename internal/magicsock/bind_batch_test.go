package magicsock

import (
	"fmt"
	"net"
	"net/netip"
	"sync"
	"testing"
	"time"

	"golang.zx2c4.com/wireguard/conn"

	"github.com/christ-pher/bnk/internal/disco"
)

func TestBatchSizeIsIdeal(t *testing.T) {
	if got := NewBind().BatchSize(); got != conn.IdealBatchSize {
		t.Errorf("BatchSize() = %d, want conn.IdealBatchSize (%d)", got, conn.IdealBatchSize)
	}
}

// receiveAll pulls packets from fn until want payloads arrived, allowing
// any batching the implementation chooses.
func receiveAll(t *testing.T, fn conn.ReceiveFunc, want int) map[string]conn.Endpoint {
	t.Helper()
	got := make(map[string]conn.Endpoint, want)
	var mu sync.Mutex
	done := make(chan struct{})
	go func() {
		defer close(done)
		batch := conn.IdealBatchSize
		packets := make([][]byte, batch)
		for i := range packets {
			packets[i] = make([]byte, 1500)
		}
		sizes := make([]int, batch)
		eps := make([]conn.Endpoint, batch)
		for {
			n, err := fn(packets, sizes, eps)
			if err != nil {
				return
			}
			mu.Lock()
			for i := 0; i < n; i++ {
				got[string(packets[i][:sizes[i]])] = eps[i]
			}
			full := len(got) >= want
			mu.Unlock()
			if full {
				return
			}
		}
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("timed out; received %d of %d packets", len(got), want)
	}
	return got
}

func TestBatchedSendDeliversAllPackets(t *testing.T) {
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
	const n = 100
	bufs := make([][]byte, n)
	for i := range bufs {
		bufs[i] = fmt.Appendf(nil, "packet-%03d", i)
	}
	if err := bindA.Send(bufs, epB); err != nil {
		t.Fatalf("Send: %v", err)
	}

	got := receiveAll(t, fnsB[0], n)
	if len(got) != n {
		t.Fatalf("received %d distinct payloads, want %d", len(got), n)
	}
	for payload, ep := range got {
		if ep.DstToString() != keyA.String() {
			t.Errorf("payload %q tagged %q, want sender %q", payload, ep.DstToString(), keyA)
		}
	}
}

func TestBatchedReceiveStillPeelsDiscoAndDropsStrangers(t *testing.T) {
	keyA, keyB := newNodeKey(t), newNodeKey(t)
	bindA, bindB := NewBind(), NewBind()
	_, addrA := openBind(t, bindA)
	fnsB, addrB := openBind(t, bindB)

	bindA.SetPeerAddr(keyB, addrB)
	bindB.SetPeerAddr(keyA, addrA)

	discoCh := make(chan []byte, 1)
	bindB.SetDiscoHandler(func(_ netip.AddrPort, pkt []byte) {
		select {
		case discoCh <- pkt:
		default:
		}
	})

	stranger, err := net.Dial("udp", addrB.String())
	if err != nil {
		t.Fatal(err)
	}
	defer stranger.Close()
	if _, err := stranger.Write([]byte("intruder-junk")); err != nil {
		t.Fatal(err)
	}
	discoPkt := append([]byte(disco.Magic), make([]byte, 60)...)
	if _, err := stranger.Write(discoPkt); err != nil {
		t.Fatal(err)
	}

	epB, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		t.Fatal(err)
	}
	if err := bindA.Send([][]byte{[]byte("legit")}, epB); err != nil {
		t.Fatalf("Send: %v", err)
	}

	got := receiveAll(t, fnsB[0], 1)
	if _, ok := got["legit"]; !ok || len(got) != 1 {
		t.Errorf("received payloads %v, want exactly [legit]", got)
	}
	select {
	case pkt := <-discoCh:
		if string(pkt[:len(disco.Magic)]) != disco.Magic {
			t.Errorf("disco handler got %q", pkt)
		}
	case <-time.After(5 * time.Second):
		t.Error("disco handler never saw the disco packet")
	}
}
