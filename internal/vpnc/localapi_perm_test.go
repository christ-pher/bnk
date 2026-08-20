package vpnc

import (
	"os"
	"path/filepath"
	"testing"
)

// The local socket must be reachable without root: parent dir 0755, socket
// itself 0666. Status/ping/netcheck are read-only diagnostics.
func TestLocalAPISocketWorldAccessible(t *testing.T) {
	sock := filepath.Join(t.TempDir(), "run", "vpn.sock")
	c := &controller{cfg: Config{Hostname: "self"}, cache: &netmapCache{}, kick: make(chan struct{}, 1)}
	ln, err := serveLocalAPI(sock, c)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
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

// A second daemon must refuse to replace a socket another daemon is
// actively serving, instead of silently hijacking its CLI.
func TestSecondDaemonCannotHijackLiveSocket(t *testing.T) {
	sock := filepath.Join(t.TempDir(), "vpn.sock")
	c := &controller{cfg: Config{Hostname: "self"}, cache: &netmapCache{}, kick: make(chan struct{}, 1)}
	ln, err := serveLocalAPI(sock, c)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	if _, err := serveLocalAPI(sock, c); err == nil {
		t.Fatal("second serveLocalAPI on a live socket succeeded, want error")
	}
}
