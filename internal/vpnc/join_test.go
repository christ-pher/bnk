package vpnc_test

import (
	"context"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/vpnc"
)

// runUnenrolled starts a daemon with no key and no server, the state a
// machine is left in by an installer that was never given one.
func runUnenrolled(t *testing.T, ctx context.Context, stateDir string) *tnetHolder {
	t.Helper()
	h := &tnetHolder{done: make(chan struct{})}
	go func() {
		defer close(h.done)
		err := vpnc.Run(ctx, vpnc.Config{
			StateDir:   stateDir,
			SocketPath: filepath.Join(stateDir, "bnk.sock"),
			Hostname:   "unsigned-in",
			CreateTUN:  h.factory,
			Logf:       t.Logf,
		})
		if err != nil && ctx.Err() == nil {
			t.Errorf("Run: %v", err)
		}
	}()
	return h
}

// The daemon must survive having no key: if it exits, nothing is left
// running for the tray to sign in through — which is exactly what left a
// double-clicked install doing nothing at all.
func TestDaemonStaysUpWhenNotEnrolled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stateDir := t.TempDir()
	runUnenrolled(t, ctx, stateDir)

	hc := statusClient(stateDir)
	deadline := time.Now().Add(10 * time.Second)
	for {
		resp, err := hc.Get("http://bnk/status")
		if err == nil {
			var st vpnc.Status
			jerr := json.NewDecoder(resp.Body).Decode(&st)
			resp.Body.Close()
			if jerr == nil {
				if st.Enrolled {
					t.Fatal("reported enrolled with no identity")
				}
				if st.Running {
					t.Fatal("reported running with no identity")
				}
				return // answering at all is the point
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("daemon never served status while unenrolled (last err: %v)", err)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// Signing in at runtime is what the tray does.
func TestJoinEnrollsARunningDaemon(t *testing.T) {
	tc := startControl(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stateDir := t.TempDir()
	h := runUnenrolled(t, ctx, stateDir)
	hc := statusClient(stateDir)

	// Wait for the local API before signing in.
	deadline := time.Now().Add(10 * time.Second)
	for {
		if resp, err := hc.Get("http://bnk/status"); err == nil {
			resp.Body.Close()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("daemon never served status")
		}
		time.Sleep(100 * time.Millisecond)
	}

	body := strings.NewReader(`{"server":"` + tc.url + `","key":"` + tc.enrollKey + `"}`)
	resp, err := hc.Post("http://bnk/join", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	msg := make([]byte, 512)
	n, _ := resp.Body.Read(msg)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("join = %d: %s", resp.StatusCode, msg[:n])
	}

	// The tunnel must actually exist afterwards, not merely be promised.
	if _, ip := h.peek(); !ip.IsValid() {
		t.Fatal("no tunnel interface after joining")
	}
	var st vpnc.Status
	if err := json.NewDecoder(mustGet(t, hc, "http://bnk/status")).Decode(&st); err != nil {
		t.Fatal(err)
	}
	if !st.Enrolled || !st.Running {
		t.Errorf("after join: enrolled=%v running=%v, want both true", st.Enrolled, st.Running)
	}
}

func TestJoinRejectsAMalformedKey(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stateDir := t.TempDir()
	runUnenrolled(t, ctx, stateDir)
	hc := statusClient(stateDir)

	deadline := time.Now().Add(10 * time.Second)
	for {
		if resp, err := hc.Get("http://bnk/status"); err == nil {
			resp.Body.Close()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("daemon never served status")
		}
		time.Sleep(100 * time.Millisecond)
	}

	resp, err := hc.Post("http://bnk/join", "application/json",
		strings.NewReader(`{"server":"https://x:1","key":"not-a-key"}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		t.Error("a malformed key was accepted")
	}
}

// mustGet returns a response body or fails the test.
func mustGet(t *testing.T, hc *http.Client, url string) *strings.Reader {
	t.Helper()
	resp, err := hc.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, 8192)
	n, _ := resp.Body.Read(buf)
	return strings.NewReader(string(buf[:n]))
}
