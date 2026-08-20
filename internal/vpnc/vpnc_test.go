package vpnc_test

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"log"
	"net/http/httptest"
	"net/netip"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"golang.zx2c4.com/wireguard/tun"
	"golang.zx2c4.com/wireguard/tun/netstack"

	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/pin"
	"vpnmesh/internal/store"
	"vpnmesh/internal/vpnc"
)

type testControl struct {
	url       string
	enrollKey string
}

// startControl runs a TLS control server with a generated pinned cert.
func startControl(t *testing.T) *testControl {
	t.Helper()
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
	secret, err := srv.NewEnrollKey()
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewUnstartedServer(srv.Handler())
	ts.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	ts.StartTLS()
	t.Cleanup(ts.Close)
	return &testControl{url: ts.URL, enrollKey: pin.FormatEnrollKey(secret, fp)}
}

// tnetOf captures the netstack.Net created for a client.
type tnetHolder struct {
	mu   sync.Mutex
	tnet *netstack.Net
	ip   netip.Addr
}

func (h *tnetHolder) factory(prefix netip.Prefix, mtu int) (tun.Device, func() error, error) {
	dev, tnet, err := netstack.CreateNetTUN([]netip.Addr{prefix.Addr()}, nil, mtu)
	if err != nil {
		return nil, nil, err
	}
	h.mu.Lock()
	h.tnet, h.ip = tnet, prefix.Addr()
	h.mu.Unlock()
	return dev, func() error { return nil }, nil
}

func (h *tnetHolder) get(t *testing.T) (*netstack.Net, netip.Addr) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		h.mu.Lock()
		tnet, ip := h.tnet, h.ip
		h.mu.Unlock()
		if tnet != nil {
			return tnet, ip
		}
		if time.Now().After(deadline) {
			t.Fatal("client never created its TUN")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func runClient(t *testing.T, ctx context.Context, tc *testControl, name, stateDir, enrollKey string) *tnetHolder {
	t.Helper()
	h := &tnetHolder{}
	go func() {
		err := vpnc.Run(ctx, vpnc.Config{
			ServerURL: tc.url,
			EnrollKey: enrollKey,
			StateDir:  stateDir,
			Hostname:  name,
			CreateTUN: h.factory,
			// log.Printf, not t.Logf: this goroutine can outlive the test.
			Logf: log.Printf,
		})
		if err != nil && ctx.Err() == nil {
			t.Errorf("%s Run: %v", name, err)
		}
	}()
	return h
}

func echoOverTunnel(t *testing.T, a, b *tnetHolder) {
	t.Helper()
	bnet, bip := b.get(t)
	anet, _ := a.get(t)

	ln, err := bnet.ListenTCPAddrPort(netip.AddrPortFrom(bip, 4242))
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

	// Generous: the full -race suite runs packages in parallel and can
	// starve handshake timers.
	deadline := time.Now().Add(45 * time.Second)
	for {
		dctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		c, err := anet.DialContextTCPAddrPort(dctx, netip.AddrPortFrom(bip, 4242))
		cancel()
		if err == nil {
			defer c.Close()
			c.Write([]byte("hi"))
			c.SetReadDeadline(time.Now().Add(10 * time.Second))
			buf := make([]byte, 64)
			n, err := c.Read(buf)
			if err != nil || string(buf[:n]) != "hi" {
				t.Fatalf("echo = %q, %v", buf[:n], err)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("tunnel never came up: %v", err)
		}
	}
}

func TestRunEstablishesTunnelBetweenTwoClients(t *testing.T) {
	tc := startControl(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	a := runClient(t, ctx, tc, "alpha", t.TempDir(), tc.enrollKey)
	b := runClient(t, ctx, tc, "beta", t.TempDir(), tc.enrollKey)
	echoOverTunnel(t, a, b)
}

func TestRunReusesStateWithoutEnrollKey(t *testing.T) {
	tc := startControl(t)
	stateA := t.TempDir()

	// First run enrolls and writes state.
	ctx1, cancel1 := context.WithCancel(context.Background())
	h := runClient(t, ctx1, tc, "alpha", stateA, tc.enrollKey)
	h.get(t)
	cancel1()

	raw, err := os.ReadFile(filepath.Join(stateA, "client.json"))
	if err != nil {
		t.Fatalf("state file after enroll: %v", err)
	}
	var st map[string]any
	if err := json.Unmarshal(raw, &st); err != nil {
		t.Fatal(err)
	}

	// Second run: no enrollment key, must come up from stored state alone.
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	a := runClient(t, ctx2, tc, "alpha", stateA, "")
	b := runClient(t, ctx2, tc, "beta", t.TempDir(), tc.enrollKey)
	echoOverTunnel(t, a, b)
}
