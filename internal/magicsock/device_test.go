package magicsock

// The Phase 0 exit criterion: two in-process wireguard-go devices, each with
// their own Bind, exchange TCP traffic over the encrypted tunnel using
// identity endpoints and a static path table. Runs without root via gvisor
// netstack TUNs.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/netip"
	"testing"
	"time"

	"golang.org/x/crypto/curve25519"
	"golang.zx2c4.com/wireguard/device"
	"golang.zx2c4.com/wireguard/tun/netstack"
)

type testNode struct {
	priv, pub [32]byte
	bind      *Bind
	dev       *device.Device
	net       *netstack.Net
	tunIP     netip.Addr
}

func newKeypair(t *testing.T) (priv, pub [32]byte) {
	t.Helper()
	if _, err := rand.Read(priv[:]); err != nil {
		t.Fatal(err)
	}
	// Standard curve25519 clamping, as WireGuard expects.
	priv[0] &= 248
	priv[31] &= 127
	priv[31] |= 64
	p, err := curve25519.X25519(priv[:], curve25519.Basepoint)
	if err != nil {
		t.Fatal(err)
	}
	copy(pub[:], p)
	return priv, pub
}

func startNode(t *testing.T, tunIP netip.Addr) *testNode {
	t.Helper()
	priv, pub := newKeypair(t)
	tunDev, tnet, err := netstack.CreateNetTUN([]netip.Addr{tunIP}, nil, 1280)
	if err != nil {
		t.Fatal(err)
	}
	bind := NewBind()
	dev := device.NewDevice(tunDev, bind, device.NewLogger(device.LogLevelError, ""))
	if err := dev.IpcSet(fmt.Sprintf("private_key=%s\n", hex.EncodeToString(priv[:]))); err != nil {
		t.Fatal(err)
	}
	if err := dev.Up(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(dev.Close)
	return &testNode{priv: priv, pub: pub, bind: bind, dev: dev, net: tnet, tunIP: tunIP}
}

// configurePeer adds peer to n's device config with an identity endpoint;
// it does NOT touch the Bind's path table.
func configurePeer(t *testing.T, n, peer *testNode) {
	t.Helper()
	cfg := fmt.Sprintf("public_key=%s\nendpoint=%s\nallowed_ip=%s/32\n",
		hex.EncodeToString(peer.pub[:]), NodeKey(peer.pub), peer.tunIP)
	if err := n.dev.IpcSet(cfg); err != nil {
		t.Fatal(err)
	}
}

// addPeer points n at peer: identity endpoint in the device config, direct
// address in the Bind's path table.
func addPeer(t *testing.T, n, peer *testNode) {
	t.Helper()
	n.bind.SetPeerAddr(NodeKey(peer.pub), netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), peer.bind.LocalPort()))
	configurePeer(t, n, peer)
}

func TestTwoDevicesExchangeTCPOverTunnel(t *testing.T) {
	a := startNode(t, netip.MustParseAddr("100.64.0.1"))
	b := startNode(t, netip.MustParseAddr("100.64.0.2"))
	addPeer(t, a, b)
	addPeer(t, b, a)
	echoTCP(t, a, b)
}

// echoTCP asserts a TCP round-trip from a to b over the tunnel.
func echoTCP(t *testing.T, a, b *testNode) {
	t.Helper()
	ln, err := b.net.ListenTCPAddrPort(netip.AddrPortFrom(b.tunIP, 7777))
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		buf := make([]byte, 64)
		n, err := c.Read(buf)
		if err != nil {
			return
		}
		c.Write(buf[:n])
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	c, err := a.net.DialContextTCPAddrPort(ctx, netip.AddrPortFrom(b.tunIP, 7777))
	if err != nil {
		t.Fatalf("dial over tunnel: %v", err)
	}
	defer c.Close()

	if _, err := c.Write([]byte("through the tunnel")); err != nil {
		t.Fatal(err)
	}
	c.SetReadDeadline(time.Now().Add(10 * time.Second))
	buf := make([]byte, 64)
	n, err := c.Read(buf)
	if err != nil {
		t.Fatalf("read echo: %v", err)
	}
	if got := string(buf[:n]); got != "through the tunnel" {
		t.Errorf("echo = %q, want %q", got, "through the tunnel")
	}
}
