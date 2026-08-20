package dataplane_test

// Isolates the TLS variable from the natlab symmetric failure: the same
// dead-endpoint relay scenario, but with the coordination channel (and so
// the relay) running over TLS with a pinned cert — exactly like production.

import (
	"context"
	"crypto/tls"
	"net/http"
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
	"vpnmesh/internal/pin"
	"vpnmesh/internal/store"
)

func TestRelayOverTLSWithDeadEndpoints(t *testing.T) {
	certPEM, keyPEM, err := pin.GenerateCert()
	if err != nil {
		t.Fatal(err)
	}
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	fp, err := pin.Fingerprint(certPEM)
	if err != nil {
		t.Fatal(err)
	}
	srv, err := server.New(store.NewFileStore(filepath.Join(t.TempDir(), "state.json")))
	if err != nil {
		t.Fatal(err)
	}
	enrollKey, err := srv.NewEnrollKey()
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewUnstartedServer(srv.Handler())
	ts.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	ts.StartTLS()
	defer ts.Close()

	tlsConf := pin.ClientTLSConfig(fp)
	hc := &http.Client{Transport: &http.Transport{TLSClientConfig: tlsConf}}

	start := func(name string, deadEp netip.AddrPort) *node {
		priv, pub := newKeypair(t)
		dPriv, dPub, err := disco.NewKeypair()
		if err != nil {
			t.Fatal(err)
		}
		resp, err := client.Enroll(context.Background(), ts.URL, hc, coord.EnrollRequest{
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
		sess, err := client.Dial(context.Background(), ts.URL, tlsConf, pub, client.Handlers{
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

	a := start("alpha", netip.MustParseAddrPort("192.0.2.10:41641"))
	b := start("beta", netip.MustParseAddrPort("192.0.2.20:41641"))

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
			c.Close()
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
			t.Fatalf("relay-over-TLS tunnel never came up: %v", err)
		}
	}
}
