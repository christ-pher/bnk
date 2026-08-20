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

func TestAdminPolicyRoundTripAndCheck(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "laptop", key32(1))
	e.enroll(t, "nas", key32(2))
	admin := httptest.NewServer(e.srv.AdminHandler("feedface"))
	defer admin.Close()

	policy := `{"rules":[{"from":["laptop"],"to":["nas"],"allow":["tcp/22"]}]}`
	req, _ := http.NewRequest(http.MethodPut, admin.URL+"/policy", strings.NewReader(policy))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("PUT /policy = %s", resp.Status)
	}

	resp, err = http.Get(admin.URL + "/policy")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got struct {
		Rules []struct {
			To []string `json:"to"`
		} `json:"rules"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got.Rules) != 1 || got.Rules[0].To[0] != "nas" {
		t.Errorf("GET /policy = %+v", got)
	}

	for query, want := range map[string]bool{
		"src=laptop&dst=nas&target=tcp/22": true,
		"src=laptop&dst=nas&target=tcp/23": false,
		"src=nas&dst=laptop&target=tcp/22": false,
	} {
		resp, err := http.Get(admin.URL + "/check?" + query)
		if err != nil {
			t.Fatal(err)
		}
		var verdict struct {
			Allowed bool `json:"allowed"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&verdict); err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if verdict.Allowed != want {
			t.Errorf("check %s = %v, want %v", query, verdict.Allowed, want)
		}
	}
}

func TestAdminPolicyRejectsInvalid(t *testing.T) {
	e := startServer(t)
	e.enroll(t, "laptop", key32(1))
	admin := httptest.NewServer(e.srv.AdminHandler("feedface"))
	defer admin.Close()

	req, _ := http.NewRequest(http.MethodPut, admin.URL+"/policy",
		strings.NewReader(`{"rules":[{"from":["ghost"],"to":["laptop"],"allow":["tcp/22"]}]}`))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		t.Error("invalid policy accepted over admin API")
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
