package server_test

// The server must drop sessions whose clients go silent (crashed machine,
// dead NAT mapping) so the mesh's online state stays truthful. Client
// keepalives are what keep a healthy-but-quiet session alive.

import (
	"testing"
	"time"

	"vpnmesh/internal/coord/client"
	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/netmap"
)

func TestServerDropsSilentSessionButKeepsKeepalivedOne(t *testing.T) {
	oldRT, oldKA := server.ReadTimeout, client.KeepaliveInterval
	server.ReadTimeout = 500 * time.Millisecond
	defer func() { server.ReadTimeout, client.KeepaliveInterval = oldRT, oldKA }()

	e := startServer(t)
	idQ, idS := ident32(t, 1), ident32(t, 2)
	e.enroll(t, "quiet", idQ.pub)
	e.enroll(t, "silent", idS.pub)

	// "quiet" keepalives normally; "silent" never sends after hello.
	client.KeepaliveInterval = 100 * time.Millisecond
	quiet, _ := dialSession(t, e, idQ.priv)
	defer quiet.Close()
	client.KeepaliveInterval = time.Hour
	silent, _ := dialSession(t, e, idS.priv)
	defer silent.Close()

	// The silent session must be dropped by the server's read deadline...
	select {
	case <-silent.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("server never dropped the silent session")
	}
	// ...while the keepalived one survives well past the timeout.
	select {
	case <-quiet.Done():
		t.Fatal("server dropped a session that was sending keepalives")
	case <-time.After(1500 * time.Millisecond):
	}
	_ = netmap.Key{}
}
