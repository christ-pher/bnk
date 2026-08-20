package server_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vpnmesh/internal/coord/server"
)

func TestAdminNodesListsEnrollmentAndOnlineState(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "alpha", key32(1))
	e.enroll(t, "beta", key32(2))
	dialSession(t, e, key32(1)) // alpha online

	admin := httptest.NewServer(e.srv.AdminHandler("feedface"))
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
	if len(nodes) != 2 {
		t.Fatalf("got %d nodes, want 2", len(nodes))
	}
	byName := map[string]server.AdminNode{}
	for _, n := range nodes {
		byName[n.Name] = n
	}
	if !byName["alpha"].Online {
		t.Error("alpha should be online (session connected)")
	}
	if byName["beta"].Online {
		t.Error("beta should be offline")
	}
	if !byName["alpha"].IP.IsValid() {
		t.Error("alpha has no IP")
	}
}

func TestAdminNewEnrollKeyReturnsFullPinnedKey(t *testing.T) {
	e := startServer(t)
	admin := httptest.NewServer(e.srv.AdminHandler("feedface"))
	defer admin.Close()

	resp, err := http.Post(admin.URL+"/enroll-keys", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Key string `json:"key"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out.Key, "vpnkey:") || !strings.HasSuffix(out.Key, ":feedface") {
		t.Errorf("key = %q, want vpnkey:<secret>:feedface", out.Key)
	}
}
