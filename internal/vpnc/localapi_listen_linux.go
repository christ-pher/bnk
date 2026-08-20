//go:build linux

package vpnc

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// serveLocalAPI exposes daemon state to the CLI over a unix socket. The
// socket is world-accessible (0666, dir 0755) so status and diagnostics
// don't need root; control verbs are gated by the peer's uid, since a
// world-accessible socket would otherwise let any local user tear the
// tunnel down. The caller owns the returned closer and must close it
// before Run returns, or a restarting daemon can have its fresh socket
// file unlinked by the old instance's teardown.
func serveLocalAPI(sock string, c *controller) (io.Closer, error) {
	if err := os.MkdirAll(filepath.Dir(sock), 0o755); err != nil {
		return nil, err
	}
	// A connectable socket means another daemon is serving it; removing
	// it would silently hijack that daemon's CLI. A stale file (daemon
	// gone, nothing accepting) refuses the dial and is safe to replace.
	if probe, err := net.DialTimeout("unix", sock, time.Second); err == nil {
		probe.Close()
		return nil, fmt.Errorf("another daemon is already serving %s", sock)
	}
	os.Remove(sock)
	ln, err := net.Listen("unix", sock)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(sock, 0o666); err != nil {
		ln.Close()
		return nil, err
	}

	// requireOwner gates control verbs: up/down must come from root or
	// the uid the daemon runs as (SO_PEERCRED).
	requireOwner := func(h http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			uid, ok := uidFromContext(r.Context())
			if !ok || (uid != 0 && uid != os.Getuid()) {
				http.Error(w, "permission denied: up/down need root (try sudo)", http.StatusForbidden)
				return
			}
			h(w, r)
		}
	}

	mux := http.NewServeMux()
	registerDiagnostics(mux, c)
	registerControl(mux, c, requireOwner)

	srv := &http.Server{
		Handler: mux,
		// Stamp each connection with the peer's uid so control verbs can
		// distinguish root/daemon-owner from other local users.
		ConnContext: func(ctx context.Context, conn net.Conn) context.Context {
			if uid, ok := peerUID(conn); ok {
				ctx = context.WithValue(ctx, peerUIDKey{}, uid)
			}
			return ctx
		},
	}
	go srv.Serve(ln)
	return ln, nil
}

type peerUIDKey struct{}

func uidFromContext(ctx context.Context) (int, bool) {
	uid, ok := ctx.Value(peerUIDKey{}).(int)
	return uid, ok
}
