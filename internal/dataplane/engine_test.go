package dataplane_test

// Phase 1 integration: two clients enroll with a real control server,
// receive netmaps over live sessions, and establish a WireGuard tunnel to
// each other driven entirely by the control plane. In-process (netstack
// TUNs, loopback endpoints), no root.

import (
	"context"
	"crypto/rand"
	"net/http/httptest"
	"net/netip"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/crypto/curve25519"
	"golang.zx2c4.com/wireguard/tun/netstack"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/coord/client"
	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/dataplane"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/store"
)

type node struct {
	engine *dataplane.Engine
	net    *netstack.Net
	ip     netip.Addr
}

func newKeypair(t *testing.T) (priv [32]byte, pub netmap.Key) {
	t.Helper()
	if _, err := rand.Read(priv[:]); err != nil {
		t.Fatal(err)
	}
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

// startClient enrolls, brings up an engine on a netstack TUN, connects the
// session, and reports its loopback endpoint.
func startClient(t *testing.T, url, enrollKey, name string) *node {
	t.Helper()
	priv, pub := newKeypair(t)
	resp, err := client.Enroll(context.Background(), url, nil, coord.EnrollRequest{
		EnrollKey: enrollKey,
		Hostname:  name,
		OS:        "test",
		NodeKey:   pub,
	})
	if err != nil {
		t.Fatalf("%s enroll: %v", name, err)
	}

	tunDev, tnet, err := netstack.CreateNetTUN([]netip.Addr{resp.IP}, nil, 1280)
	if err != nil {
		t.Fatal(err)
	}
	engine, err := dataplane.New(tunDev, priv)
	if err != nil {
		t.Fatalf("%s engine: %v", name, err)
	}
	t.Cleanup(engine.Close)

	sess, err := client.Dial(context.Background(), url, nil, pub, client.Handlers{
		OnNetmap: func(nm netmap.Netmap) {
			if err := engine.ApplyNetmap(nm); err != nil {
				t.Errorf("%s apply netmap: %v", name, err)
			}
		},
	})
	if err != nil {
		t.Fatalf("%s dial: %v", name, err)
	}
	t.Cleanup(sess.Close)

	self := netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), engine.LocalPort())
	if err := sess.SendEndpoints([]netip.AddrPort{self}); err != nil {
		t.Fatal(err)
	}
	return &node{engine: engine, net: tnet, ip: resp.IP}
}

func TestControlPlaneEstablishesTunnel(t *testing.T) {
	srv, err := server.New(store.NewFileStore(filepath.Join(t.TempDir(), "state.json")))
	if err != nil {
		t.Fatal(err)
	}
	enrollKey, err := srv.NewEnrollKey()
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	a := startClient(t, ts.URL, enrollKey, "alpha")
	b := startClient(t, ts.URL, enrollKey, "beta")

	ln, err := b.net.ListenTCPAddrPort(netip.AddrPortFrom(b.ip, 9999))
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				buf := make([]byte, 64)
				n, err := c.Read(buf)
				if err == nil {
					c.Write(buf[:n])
				}
			}()
		}
	}()

	// Netmaps propagate asynchronously; retry the dial until the tunnel is up.
	deadline := time.Now().Add(20 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		c, err := a.net.DialContextTCPAddrPort(ctx, netip.AddrPortFrom(b.ip, 9999))
		cancel()
		if err == nil {
			defer c.Close()
			if _, err := c.Write([]byte("via control plane")); err != nil {
				t.Fatal(err)
			}
			c.SetReadDeadline(time.Now().Add(10 * time.Second))
			buf := make([]byte, 64)
			n, err := c.Read(buf)
			if err != nil {
				t.Fatalf("read echo: %v", err)
			}
			if got := string(buf[:n]); got != "via control plane" {
				t.Errorf("echo = %q", got)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("tunnel never came up: %v", err)
		}
	}
}
