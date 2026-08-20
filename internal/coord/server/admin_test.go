package server_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/coord/server"
)

func TestAdminNodesListsEnrollmentAndOnlineState(t *testing.T) {
	e := startServer(t)
	id := ident32(t, 1)
	e.enroll(t, "alpha", id.pub)
	e.enroll(t, "beta", key32(2))
	dialSession(t, e, id.priv) // alpha online

	admin := httptest.NewServer(e.srv.AdminHandler("feedface", ""))
	defer admin.Close()

	// Session registration completes shortly after Dial returns (the
	// server verifies the key proof first), so poll for alpha online.
	var byName map[string]server.AdminNode
	deadline := time.Now().Add(5 * time.Second)
	for {
		resp, err := http.Get(admin.URL + "/nodes")
		if err != nil {
			t.Fatal(err)
		}
		var nodes []server.AdminNode
		err = json.NewDecoder(resp.Body).Decode(&nodes)
		resp.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if len(nodes) != 2 {
			t.Fatalf("got %d nodes, want 2", len(nodes))
		}
		byName = map[string]server.AdminNode{}
		for _, n := range nodes {
			byName[n.Name] = n
		}
		if byName["alpha"].Online {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("alpha never came online (session connected)")
		}
		time.Sleep(50 * time.Millisecond)
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
	admin := httptest.NewServer(e.srv.AdminHandler("feedface", ""))
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
	admin := httptest.NewServer(e.srv.AdminHandler("feedface", ""))
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

func TestAdminKeyLifecycle(t *testing.T) {
	e := startServer(t)
	admin := httptest.NewServer(e.srv.AdminHandler("feedface", ""))
	defer admin.Close()

	// Default mint: one-time. With ?reusable=true: reusable.
	resp, err := http.Post(admin.URL+"/enroll-keys?reusable=true&ttl=1h", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	// startServer minted a baseline key; ours is the newest entry.
	resp, err = http.Get(admin.URL + "/enroll-keys")
	if err != nil {
		t.Fatal(err)
	}
	var keys []server.AdminKey
	if err := json.NewDecoder(resp.Body).Decode(&keys); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	mine := keys[len(keys)-1]
	if !mine.Reusable || mine.Revoked {
		t.Fatalf("minted key = %+v, want reusable unrevoked", mine)
	}

	resp, err = http.Post(admin.URL+"/enroll-keys/revoke?prefix="+mine.Prefix, "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("revoke = %s", resp.Status)
	}

	resp, _ = http.Get(admin.URL + "/enroll-keys")
	keys = nil
	json.NewDecoder(resp.Body).Decode(&keys)
	resp.Body.Close()
	if got := keys[len(keys)-1]; !got.Revoked {
		t.Fatalf("after revoke: %+v", got)
	}
}

func TestAdminMintDefaultsToOneTime(t *testing.T) {
	e := startServer(t)
	admin := httptest.NewServer(e.srv.AdminHandler("feedface", ""))
	defer admin.Close()

	resp, err := http.Post(admin.URL+"/enroll-keys", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	keys := e.srv.EnrollKeys()
	if got := keys[len(keys)-1]; got.Reusable {
		t.Fatalf("default-minted key = %+v, want one-time", got)
	}
}

func TestAdminNewEnrollKeyReturnsFullPinnedKey(t *testing.T) {
	e := startServer(t)
	admin := httptest.NewServer(e.srv.AdminHandler("feedface", ""))
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
	if !strings.HasPrefix(out.Key, "bnkkey:") || !strings.HasSuffix(out.Key, ":feedface") {
		t.Errorf("key = %q, want bnkkey:<secret>:feedface", out.Key)
	}
}

func TestAdminInfoReportsPublicURL(t *testing.T) {
	e := startServer(t)
	admin := httptest.NewServer(e.srv.AdminHandler("feedface", "https://203.0.113.7:8443"))
	defer admin.Close()

	resp, err := http.Get(admin.URL + "/info")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var info struct {
		PublicURL string `json:"public_url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		t.Fatal(err)
	}
	if info.PublicURL != "https://203.0.113.7:8443" {
		t.Errorf("public_url = %q, want the configured URL", info.PublicURL)
	}
}
