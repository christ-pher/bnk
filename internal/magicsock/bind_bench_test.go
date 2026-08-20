package magicsock

import (
	"testing"

	"golang.zx2c4.com/wireguard/conn"
)

// BenchmarkBindThroughput pushes full batches through two binds over
// loopback: the send side exercises sendmmsg, the receive side recvmmsg.
// Compare ns/op across changes to the batch plumbing.
func BenchmarkBindThroughput(b *testing.B) {
	keyA, keyB := newNodeKey(b), newNodeKey(b)
	bindA, bindB := NewBind(), NewBind()
	_, addrA := openBind(b, bindA)
	fnsB, addrB := openBind(b, bindB)
	bindA.SetPeerAddr(keyB, addrB)
	bindB.SetPeerAddr(keyA, addrA)
	ep, err := bindA.ParseEndpoint(keyB.String())
	if err != nil {
		b.Fatal(err)
	}

	const pktSize = 1280
	batch := conn.IdealBatchSize
	bufs := make([][]byte, batch)
	for i := range bufs {
		bufs[i] = make([]byte, pktSize)
		bufs[i][0] = 4 // WireGuard data-packet type: not disco, not STUN
	}

	// Drain receiver: loopback send blocks once the socket buffer fills,
	// so the sender measures the full send+receive pipeline.
	go func() {
		packets := make([][]byte, batch)
		for i := range packets {
			packets[i] = make([]byte, 1500)
		}
		sizes := make([]int, batch)
		eps := make([]conn.Endpoint, batch)
		for {
			if _, err := fnsB[0](packets, sizes, eps); err != nil {
				return
			}
		}
	}()

	b.SetBytes(int64(batch * pktSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := bindA.Send(bufs, ep); err != nil {
			b.Fatal(err)
		}
	}
}
