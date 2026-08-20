package vpnc_test

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/vpnc"
)

func postLocal(t *testing.T, hc *http.Client, path string) int {
	t.Helper()
	resp, err := hc.Post("http://bnk"+path, "application/json", nil)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

// waitRunning polls /status until it reports the wanted running state.
func waitRunning(t *testing.T, hc *http.Client, want bool) vpnc.Status {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for {
		resp, err := hc.Get("http://bnk/status")
		if err == nil {
			var st vpnc.Status
			jerr := json.NewDecoder(resp.Body).Decode(&st)
			resp.Body.Close()
			if jerr == nil && st.Running == want {
				return st
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("status never reached running=%v (last err: %v)", want, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func TestDownAndUpOverLocalAPI(t *testing.T) {
	tc := startControl(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stateA := t.TempDir()
	a := runClient(t, ctx, tc, "alpha", stateA, tc.enrollKey)
	b := runClient(t, ctx, tc, "beta", t.TempDir(), tc.enrollKey)
	echoOverTunnel(t, a, b)

	hc := statusClient(stateA)
	if code := postLocal(t, hc, "/down"); code != http.StatusOK {
		t.Fatalf("down = %d", code)
	}
	waitRunning(t, hc, false)

	// Diagnostics refuse while down instead of hanging.
	resp, err := hc.Get("http://bnk/ping?peer=beta")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("ping while down = %d, want 503", resp.StatusCode)
	}

	if code := postLocal(t, hc, "/up"); code != http.StatusOK {
		t.Fatalf("up = %d", code)
	}
	waitRunning(t, hc, true)
	// b's echo listener survived; a's rebuilt tunnel must reach it again.
	dialEcho(t, a, b)
}

func TestDownStatePersistsAcrossRestart(t *testing.T) {
	tc := startControl(t)
	stateA := t.TempDir()

	ctx1, cancel1 := context.WithCancel(context.Background())
	h := runClient(t, ctx1, tc, "alpha", stateA, tc.enrollKey)
	h.get(t)
	hc := statusClient(stateA)
	waitRunning(t, hc, true)
	if code := postLocal(t, hc, "/down"); code != http.StatusOK {
		t.Fatalf("down = %d", code)
	}
	waitRunning(t, hc, false)
	cancel1()
	<-h.done // the old daemon must release the socket before the restart

	// Restart the daemon: it must come back with the tunnel still down.
	// Fresh client: hc's pooled keep-alive conn still points at the old
	// (unlinked) socket.
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	h2 := runClient(t, ctx2, tc, "alpha", stateA, "")
	hc = statusClient(stateA)
	waitRunning(t, hc, false)
	time.Sleep(500 * time.Millisecond)
	if tnet, _ := h2.peek(); tnet != nil {
		t.Fatal("daemon restarted with tunnel up despite persisted down state")
	}

	// /up on the restarted daemon brings the tunnel back. Retry: the old
	// listener's teardown races the new daemon's socket takeover.
	deadline := time.Now().Add(15 * time.Second)
	for {
		resp, err := hc.Post("http://bnk/up", "application/json", nil)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("up after restart never succeeded (last err: %v)", err)
		}
		time.Sleep(200 * time.Millisecond)
	}
	h2.get(t)
	waitRunning(t, hc, true)
}
