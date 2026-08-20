package server_test

// Enrolling requires an enrollment key, but a session must additionally
// PROVE it holds the node's private key: knowledge of an enrolled public
// key alone must not admit a session.

import (
	"bufio"
	"fmt"
	"net"
	"net/url"
	"testing"
	"time"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/netmap"
)

// rawHello connects to /c, upgrades, and sends a hello claiming pub —
// then answers any challenge with garbage, like an impersonator would.
func rawHello(t *testing.T, e *env, pub netmap.Key) net.Conn {
	t.Helper()
	u, err := url.Parse(e.ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	conn, err := net.Dial("tcp", u.Host)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	fmt.Fprintf(conn, "GET /c HTTP/1.1\r\nHost: %s\r\nConnection: Upgrade\r\nUpgrade: vpn-coord/1\r\n\r\n", u.Host)
	br := bufio.NewReader(conn)
	// Skip HTTP 101 headers.
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		if line == "\r\n" {
			break
		}
	}
	hello, _ := coord.EncodeControl(coord.Envelope{T: coord.MsgHello, Hello: &coord.Hello{NodeKey: pub}})
	if err := coord.WriteFrame(conn, coord.FrameControl, hello); err != nil {
		t.Fatal(err)
	}
	// Answer whatever comes next with garbage "auth".
	go func() {
		coord.ReadFrame(br)
		auth, _ := coord.EncodeControl(coord.Envelope{T: coord.MsgAuth, Auth: &coord.Auth{Sealed: []byte("not a real proof")}})
		coord.WriteFrame(conn, coord.FrameControl, auth)
	}()
	return conn
}

func TestSessionWithoutPrivateKeyIsRejected(t *testing.T) {
	e := startServer(t)
	id := ident(t)
	e.enroll(t, "victim", id.pub)

	rawHello(t, e, id.pub)

	// The impersonator must never appear online.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		for _, n := range e.srv.EnrollKeys() {
			_ = n
		}
		online := false
		for _, an := range adminNodes(t, e) {
			if an.Name == "victim" && an.Online {
				online = true
			}
		}
		if online {
			t.Fatal("session admitted without proof of the private key")
		}
		time.Sleep(100 * time.Millisecond)
	}
}
