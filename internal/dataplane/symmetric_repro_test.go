package dataplane_test

// Reproduction for the natlab symmetric-NAT failure: disco-enabled clients
// whose advertised endpoints are all unreachable (as behind fully-random
// NATs) must still get a working tunnel via the relay. In the lab the
// tunnel never came up at all.

import (
	"context"
	"net/http/httptest"
	"net/netip"
	"path/filepath"
	"testing"
	"time"

	"golang.zx2c4.com/wireguard/tun/netstack"

	"github.com/christ-pher/bnk/internal/coord"
	"github.com/christ-pher/bnk/internal/coord/client"
	"github.com/christ-pher/bnk/internal/coord/server"
	"github.com/christ-pher/bnk/internal/dataplane"
	"github.com/christ-pher/bnk/internal/disco"
	"github.com/christ-pher/bnk/internal/netmap"
	"github.com/christ-pher/bnk/internal/store"
)

// startDeadEndClient is startDiscoClient except every advertised endpoint
// is unreachable (TEST-NET-1 space), so hole punching can never succeed.
func startDeadEndClient(t *testing.T, url, enrollKey, name string, deadEp netip.AddrPort) *node {
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

	sess, err := client.Dial(context.Background(), url, nil, netmap.Key(priv), client.Handlers{
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
	engine.SetSelfEndpoints([]netip.AddrPort{deadEp})
	if err := sess.SendEndpoints([]netip.AddrPort{deadEp}); err != nil {
		t.Fatal(err)
	}
	return &node{engine: engine, net: tnet, ip: resp.IP, sess: sess}
}

func TestRelayStillWorksWhenAllAdvertisedEndpointsAreDead(t *testing.T) {
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

	a := startDeadEndClient(t, ts.URL, enrollKey, "alpha", netip.MustParseAddrPort("192.0.2.10:41641"))
	b := startDeadEndClient(t, ts.URL, enrollKey, "beta", netip.MustParseAddrPort("192.0.2.20:41641"))

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

	deadline := time.Now().Add(25 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		c, err := a.net.DialContextTCPAddrPort(ctx, netip.AddrPortFrom(b.ip, 9999))
		cancel()
		if err == nil {
			c.Close()
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("relay tunnel never came up with dead advertised endpoints: %v", err)
		}
	}
}

func TestPingPeerSucceedsAfterUpgrade(t *testing.T) {
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
	_ = b

	// Wait for the direct path, then a disco ping must succeed too.
	deadline := time.Now().Add(20 * time.Second)
	for {
		var direct bool
		var peerKey [32]byte
		for _, pp := range a.engine.PeerPaths() {
			if pp.Direct {
				direct, peerKey = true, pp.Key
			}
		}
		if direct {
			res, err := a.engine.PingPeer(peerKey, 5*time.Second)
			if err != nil {
				t.Fatalf("PingPeer on a proven-direct peer failed: %v", err)
			}
			if res.RTT < 0 {
				t.Errorf("rtt = %v", res.RTT)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("path never became direct")
		}
		time.Sleep(200 * time.Millisecond)
	}
}
