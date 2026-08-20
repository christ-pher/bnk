package selfupdate_test

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/christ-pher/bnk/internal/selfupdate"
)

// releaseServer serves a fake GitHub releases surface: /releases/latest
// redirects to the tag page, and download URLs serve the binary and its
// SHA256SUMS.
func releaseServer(t *testing.T, tag string, binary []byte) *httptest.Server {
	t.Helper()
	asset := fmt.Sprintf("tool-linux-%s", runtime.GOARCH)
	sums := fmt.Sprintf("%x  %s\n", sha256.Sum256(binary), asset)
	mux := http.NewServeMux()
	mux.HandleFunc("/o/r/releases/latest", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/o/r/releases/tag/"+tag, http.StatusFound)
	})
	mux.HandleFunc("/o/r/releases/download/"+tag+"/"+asset, func(w http.ResponseWriter, r *http.Request) {
		w.Write(binary)
	})
	mux.HandleFunc("/o/r/releases/download/"+tag+"/SHA256SUMS", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, sums)
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

func TestRunReplacesBinaryWhenNewer(t *testing.T) {
	newBin := []byte("#!new binary v2")
	ts := releaseServer(t, "v0.2.0", newBin)

	target := filepath.Join(t.TempDir(), "tool")
	if err := os.WriteFile(target, []byte("old"), 0o755); err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	err := selfupdate.Run(selfupdate.Config{
		BaseURL: ts.URL + "/o/r",
		Asset:   "tool",
		Version: "v0.1.0",
		Target:  target,
		Out:     &out,
	})
	if err != nil {
		t.Fatalf("Run: %v (output: %s)", err, out.String())
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(newBin) {
		t.Errorf("target = %q, want the new binary", got)
	}
	fi, _ := os.Stat(target)
	if fi.Mode().Perm() != 0o755 {
		t.Errorf("target mode = %o, want 755", fi.Mode().Perm())
	}
	if !strings.Contains(out.String(), "v0.1.0") || !strings.Contains(out.String(), "v0.2.0") {
		t.Errorf("output should mention old and new versions: %q", out.String())
	}
}

func TestRunNoopWhenAlreadyLatest(t *testing.T) {
	ts := releaseServer(t, "v0.2.0", []byte("bin"))
	target := filepath.Join(t.TempDir(), "tool")
	if err := os.WriteFile(target, []byte("current"), 0o755); err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	err := selfupdate.Run(selfupdate.Config{
		BaseURL: ts.URL + "/o/r",
		Asset:   "tool",
		Version: "v0.2.0",
		Target:  target,
		Out:     &out,
	})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(target)
	if string(got) != "current" {
		t.Error("binary replaced despite being up to date")
	}
	if !strings.Contains(out.String(), "up to date") {
		t.Errorf("output = %q, want an up-to-date notice", out.String())
	}
}

func TestRunRejectsChecksumMismatch(t *testing.T) {
	// A server whose download disagrees with its SHA256SUMS.
	asset := fmt.Sprintf("tool-linux-%s", runtime.GOARCH)
	mux := http.NewServeMux()
	mux.HandleFunc("/o/r/releases/latest", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/o/r/releases/tag/v0.2.0", http.StatusFound)
	})
	mux.HandleFunc("/o/r/releases/download/v0.2.0/"+asset, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("evil binary"))
	})
	mux.HandleFunc("/o/r/releases/download/v0.2.0/SHA256SUMS", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "%x  %s\n", sha256.Sum256([]byte("good binary")), asset)
	})
	bad := httptest.NewServer(mux)
	defer bad.Close()

	target := filepath.Join(t.TempDir(), "tool")
	if err := os.WriteFile(target, []byte("old"), 0o755); err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	err := selfupdate.Run(selfupdate.Config{
		BaseURL: bad.URL + "/o/r",
		Asset:   "tool",
		Version: "v0.1.0",
		Target:  target,
		Out:     &out,
	})
	if err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("err = %v, want checksum mismatch", err)
	}
	got, _ := os.ReadFile(target)
	if string(got) != "old" {
		t.Error("binary was replaced despite checksum mismatch")
	}
}
