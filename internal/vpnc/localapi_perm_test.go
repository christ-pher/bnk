package vpnc

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// The local socket must be reachable without root: parent dir 0755, socket
// itself 0666. Status/ping/netcheck are read-only diagnostics.
func TestLocalAPISocketWorldAccessible(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sock := filepath.Join(t.TempDir(), "run", "vpn.sock")
	if err := serveLocalAPI(ctx, sock, "self", &netmapCache{}, nil, ""); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(sock)
	if err != nil {
		t.Fatal(err)
	}
	if perm := fi.Mode().Perm(); perm != 0o666 {
		t.Errorf("socket mode = %o, want 666", perm)
	}
	di, err := os.Stat(filepath.Dir(sock))
	if err != nil {
		t.Fatal(err)
	}
	if perm := di.Mode().Perm(); perm != 0o755 {
		t.Errorf("socket dir mode = %o, want 755", perm)
	}
}
