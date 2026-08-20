// Package stunner provides both halves of endpoint discovery: a STUN
// responder the control server hosts on its UDP port, and a client that
// queries through the magicsock Bind so the discovered mapping matches the
// WireGuard socket.
package stunner

import (
	"context"
	"fmt"
	"net"
	"net/netip"
	"sync"
	"time"

	"github.com/pion/stun/v3"

	"github.com/christ-pher/bnk/internal/magicsock"
)

// Serve answers STUN binding requests on pc until ctx is canceled.
func Serve(ctx context.Context, pc net.PacketConn) error {
	go func() {
		<-ctx.Done()
		pc.Close()
	}()
	buf := make([]byte, 1500)
	for {
		n, src, err := pc.ReadFrom(buf)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		req := &stun.Message{Raw: append([]byte(nil), buf[:n]...)}
		if req.Decode() != nil || req.Type != stun.BindingRequest {
			continue
		}
		udpSrc, ok := src.(*net.UDPAddr)
		if !ok {
			continue
		}
		resp, err := stun.Build(
			stun.NewTransactionIDSetter(req.TransactionID),
			stun.BindingSuccess,
			&stun.XORMappedAddress{IP: udpSrc.IP, Port: udpSrc.Port},
			stun.Fingerprint,
		)
		if err != nil {
			continue
		}
		pc.WriteTo(resp.Raw, src)
	}
}

// Client sends binding requests through a Bind and matches responses the
// Bind's demux hands back.
type Client struct {
	bind    *magicsock.Bind
	mu      sync.Mutex
	pending map[[12]byte]chan netip.AddrPort
}

func NewClient(bind *magicsock.Bind) *Client {
	c := &Client{bind: bind, pending: make(map[[12]byte]chan netip.AddrPort)}
	bind.SetSTUNHandler(c.handleResponse)
	return c
}

func (c *Client) handleResponse(pkt []byte) {
	m := &stun.Message{Raw: pkt}
	if m.Decode() != nil {
		return
	}
	var xor stun.XORMappedAddress
	if xor.GetFrom(m) != nil {
		return
	}
	addr, ok := netip.AddrFromSlice(xor.IP)
	if !ok {
		return
	}
	c.mu.Lock()
	ch, exists := c.pending[m.TransactionID]
	c.mu.Unlock()
	if exists {
		select {
		case ch <- netip.AddrPortFrom(addr.Unmap(), uint16(xor.Port)):
		default:
		}
	}
}

// Query asks server for our reflexive address as seen from the Bind's
// socket, retransmitting until ctx expires.
func (c *Client) Query(ctx context.Context, server netip.AddrPort) (netip.AddrPort, error) {
	req, err := stun.Build(stun.TransactionID, stun.BindingRequest)
	if err != nil {
		return netip.AddrPort{}, err
	}
	ch := make(chan netip.AddrPort, 1)
	c.mu.Lock()
	c.pending[req.TransactionID] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.pending, req.TransactionID)
		c.mu.Unlock()
	}()

	retry := time.NewTicker(800 * time.Millisecond)
	defer retry.Stop()
	for {
		if err := c.bind.SendRaw(server, req.Raw); err != nil {
			return netip.AddrPort{}, err
		}
		select {
		case <-ctx.Done():
			return netip.AddrPort{}, fmt.Errorf("stun query to %v: %w", server, ctx.Err())
		case observed := <-ch:
			return observed, nil
		case <-retry.C:
		}
	}
}
