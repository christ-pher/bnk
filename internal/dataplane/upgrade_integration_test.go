package dataplane_test

// Phase 4 exit criterion (in-process): peers that start on the relay
// discover a direct path via disco (call-me-maybe through the server,
// ping/pong over UDP) and upgrade — while the tunnel keeps working.

import (
	"context"
	"net/http/httptest"
	"net/netip"
	"path/filepath"
	"testing"
	"time"

	"golang.zx2c4.com/wireguard/tun/netstack"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/coord/client"
	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/dataplane"
	"vpnmesh/internal/disco"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/store"
)

// startDiscoClient wires the full Phase 4 client: disco keys at enroll,
// disco_fwd both ways, and loopback self-endpoints for the path manager —
// but no blind endpoint advertisement, so all traffic starts on the relay.
func startDiscoClient(t *testing.T, url, enrollKey, name string) *node {
	t.Helper()
	priv, pub := newKeypair(t)
	dPriv, dPub, err := disco.NewKeypair()
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Enroll(context.Background(), url, nil, coord.EnrollRequest{
		EnrollKey: enrollKey, Hostname: name, OS: "test", NodeKey: pub, DiscoKey: dPub,
	})
	if err != nil {
		t.Fatalf("%s enroll: %v", name, err)
	}
	tunDev, tnet, err := netstack.CreateNetTUN([]netip.Addr{resp.IP}, nil, 1280)
	if err != nil {
		t.Fatal(err)
	}
	engine, err := dataplane.New(tunDev, priv, dPriv, dPub)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(engine.Close)

	sess, err := client.Dial(context.Background(), url, nil, pub, client.Handlers{
		OnNetmap: func(nm netmap.Netmap) { _ = engine.ApplyNetmap(nm) },
		OnRelayData: func(src netmap.NodeID, pkt []byte) {
			engine.DeliverRelay(uint32(src), pkt)
		},
		OnDiscoFwd: func(src netmap.NodeID, payload []byte) {
			engine.HandleDiscoFwd(payload)
		},
	})
	if err != nil {
		t.Fatalf("%s dial: %v", name, err)
	}
	t.Cleanup(func() {
		sess.Close()
		<-sess.Done()
	})
	engine.SetRelaySender(func(dst uint32, pkt []byte) error {
		return sess.SendRelay(netmap.NodeID(dst), pkt)
	})
	engine.SetDiscoFwdSender(func(dst uint32, payload []byte) error {
		return sess.SendDiscoFwd(netmap.NodeID(dst), payload)
	})
	engine.SetSelfEndpoints([]netip.AddrPort{
		netip.AddrPortFrom(netip.MustParseAddr("127.0.0.1"), engine.LocalPort()),
	})
	return &node{engine: engine, net: tnet, ip: resp.IP, sess: sess}
}

func TestRelayedPeersUpgradeToDirectViaDisco(t *testing.T) {
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

	a := startDiscoClient(t, ts.URL, enrollKey, "alpha")
	b := startDiscoClient(t, ts.URL, enrollKey, "beta")

	// Tunnel first comes up over the relay (no endpoints were advertised
	// in the netmap), then disco must upgrade it to direct.
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
				if n, err := c.Read(buf); err == nil {
					c.Write(buf[:n])
				}
			}()
		}
	}()

	echo := func() bool {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		c, err := a.net.DialContextTCPAddrPort(ctx, netip.AddrPortFrom(b.ip, 9999))
		if err != nil {
			return false
		}
		defer c.Close()
		c.Write([]byte("hi"))
		c.SetReadDeadline(time.Now().Add(3 * time.Second))
		buf := make([]byte, 8)
		n, err := c.Read(buf)
		return err == nil && string(buf[:n]) == "hi"
	}

	bothDirect := func() bool {
		for _, n := range []*node{a, b} {
			direct := false
			for _, pp := range n.engine.PeerPaths() {
				if pp.Direct {
					direct = true
				}
			}
			if !direct {
				return false
			}
		}
		return true
	}

	deadline := time.Now().Add(30 * time.Second)
	for {
		if echo() && bothDirect() {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("upgrade never happened: echo=%v bothDirect=%v", echo(), bothDirect())
		}
		time.Sleep(500 * time.Millisecond)
	}
}
