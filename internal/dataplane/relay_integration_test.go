package dataplane_test

// Phase 2 exit criterion: when peers share no usable endpoints (both
// "behind NAT"), the tunnel still comes up by relaying encrypted packets
// through the control server.

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
	"github.com/christ-pher/bnk/internal/netmap"
	"github.com/christ-pher/bnk/internal/store"
)

// startRelayOnlyClient is startClient with one difference: it never
// reports endpoints, so peers can only reach it through the relay.
func startRelayOnlyClient(t *testing.T, url, enrollKey, name string) *node {
	t.Helper()
	priv, pub := newKeypair(t)
	resp, err := client.Enroll(context.Background(), url, nil, coord.EnrollRequest{
		EnrollKey: enrollKey, Hostname: name, OS: "test", NodeKey: pub,
	})
	if err != nil {
		t.Fatalf("%s enroll: %v", name, err)
	}
	tunDev, tnet, err := netstack.CreateNetTUN([]netip.Addr{resp.IP}, nil, 1280)
	if err != nil {
		t.Fatal(err)
	}
	engine, err := dataplane.New(tunDev, priv, [32]byte{}, [32]byte{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(engine.Close)

	sess, err := client.Dial(context.Background(), url, nil, netmap.Key(priv), client.Handlers{
		OnNetmap: func(nm netmap.Netmap) { _ = engine.ApplyNetmap(nm) },
		OnRelayData: func(src netmap.NodeID, pkt []byte) {
			engine.DeliverRelay(uint32(src), pkt)
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
	return &node{engine: engine, net: tnet, ip: resp.IP, sess: sess}
}

func TestTunnelComesUpViaRelayWhenPeersShareNoEndpoints(t *testing.T) {
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

	a := startRelayOnlyClient(t, ts.URL, enrollKey, "alpha")
	b := startRelayOnlyClient(t, ts.URL, enrollKey, "beta")

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

	deadline := time.Now().Add(20 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		c, err := a.net.DialContextTCPAddrPort(ctx, netip.AddrPortFrom(b.ip, 9999))
		cancel()
		if err == nil {
			defer c.Close()
			c.Write([]byte("relayed"))
			c.SetReadDeadline(time.Now().Add(10 * time.Second))
			buf := make([]byte, 64)
			n, err := c.Read(buf)
			if err != nil || string(buf[:n]) != "relayed" {
				t.Fatalf("echo = %q, %v", buf[:n], err)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("relay tunnel never came up: %v", err)
		}
	}
}
