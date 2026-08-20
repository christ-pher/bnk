package stunner_test

import (
	"context"
	"net"
	"net/netip"
	"testing"
	"time"

	"golang.zx2c4.com/wireguard/conn"

	"vpnmesh/internal/magicsock"
	"vpnmesh/internal/stunner"
)

// pumpAll drives the Bind's ReceiveFuncs so its demux runs, discarding
// surfaced WireGuard packets.
func pumpAll(fns []conn.ReceiveFunc) {
	for _, fn := range fns {
		go func(fn conn.ReceiveFunc) {
			for {
				packets := [][]byte{make([]byte, 1500)}
				sizes := make([]int, 1)
				eps := make([]conn.Endpoint, 1)
				if _, err := fn(packets, sizes, eps); err != nil {
					return
				}
			}
		}(fn)
	}
}

// The client queries through a magicsock Bind (the same socket WireGuard
// uses, so the discovered mapping matches the WG port) against our own
// STUN responder. On loopback the "reflexive" address is simply the
// Bind's local address — which is exactly what the test can verify.
func TestQueryThroughBindReturnsObservedAddress(t *testing.T) {
	// STUN server on a loopback UDP socket.
	pc, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer pc.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go stunner.Serve(ctx, pc)
	serverAddr := netip.MustParseAddrPort(pc.LocalAddr().String())

	// Client bind, pumped like wireguard-go would.
	bind := magicsock.NewBind()
	fns, port, err := bind.Open(0)
	if err != nil {
		t.Fatal(err)
	}
	defer bind.Close()
	pumpAll(fns)

	client := stunner.NewClient(bind)
	qctx, qcancel := context.WithTimeout(ctx, 5*time.Second)
	defer qcancel()
	observed, err := client.Query(qctx, serverAddr)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if observed.Port() != port {
		t.Errorf("observed = %v, want port %d (the bind's own port on loopback)", observed, port)
	}
	if !observed.Addr().IsLoopback() {
		t.Errorf("observed addr = %v, want loopback", observed.Addr())
	}
}
