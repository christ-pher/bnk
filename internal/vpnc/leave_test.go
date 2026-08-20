package vpnc_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/christ-pher/bnk/internal/coord/server"
	"github.com/christ-pher/bnk/internal/vpnc"
)

func serverNodeNames(t *testing.T, tc *testControl) map[string]bool {
	t.Helper()
	admin := httptest.NewServer(tc.srv.AdminHandler("fp", ""))
	defer admin.Close()
	resp, err := http.Get(admin.URL + "/nodes")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var nodes []server.AdminNode
	if err := json.NewDecoder(resp.Body).Decode(&nodes); err != nil {
		t.Fatal(err)
	}
	out := map[string]bool{}
	for _, n := range nodes {
		out[n.Name] = true
	}
	return out
}

// This is the path the uninstallers take: a node that leaves must
// disappear from the server rather than lingering as a stale entry.
func TestLeaveDeregistersTheNode(t *testing.T) {
	tc := startControl(t)
	ctx, cancel := context.WithCancel(context.Background())

	stateA := t.TempDir()
	a := runClient(t, ctx, tc, "alpha", stateA, tc.enrollKey)
	a.get(t)
	if !serverNodeNames(t, tc)["alpha"] {
		t.Fatal("alpha never enrolled")
	}

	// Uninstall stops the daemon, then deregisters using stored state.
	cancel()
	<-a.done

	if err := vpnc.Leave(context.Background(), stateA); err != nil {
		t.Fatalf("Leave: %v", err)
	}
	if serverNodeNames(t, tc)["alpha"] {
		t.Error("alpha is still registered after leaving")
	}
}

func TestLeaveWithoutEnrollmentIsAnError(t *testing.T) {
	if err := vpnc.Leave(context.Background(), t.TempDir()); err == nil {
		t.Error("expected an error leaving from a state dir with no identity")
	}
}
