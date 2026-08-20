package client

// A silently-dead coordination connection (VPS reboot, middlebox drop)
// must be detected by the read deadline so the reconnect loop can fire —
// not hang forever.

import (
	"bufio"
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/netmap"
)

func TestSessionDiesWhenServerGoesSilent(t *testing.T) {
	oldKA, oldRT := KeepaliveInterval, ReadTimeout
	KeepaliveInterval, ReadTimeout = 50*time.Millisecond, 300*time.Millisecond
	defer func() { KeepaliveInterval, ReadTimeout = oldKA, oldRT }()

	// A server that upgrades, swallows the hello, then goes silent forever.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, bufrw, err := w.(http.Hijacker).Hijack()
		if err != nil {
			return
		}
		conn.Write([]byte("HTTP/1.1 101 Switching Protocols\r\nUpgrade: vpn-coord/1\r\nConnection: Upgrade\r\n\r\n"))
		coord.ReadFrame(bufrw.Reader)
		// Keep the TCP connection open but never send another byte, and
		// keep draining so client keepalives don't block.
		go func() {
			br := bufio.NewReader(conn)
			for {
				if _, _, err := coord.ReadFrame(br); err != nil {
					return
				}
			}
		}()
		<-r.Context().Done()
		_ = conn
	}))
	defer ts.Close()

	sess, err := Dial(context.Background(), ts.URL, nil, netmap.Key{1}, Handlers{})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Close()

	select {
	case <-sess.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("session never noticed the server went silent")
	}
	_ = net.ErrClosed
}
