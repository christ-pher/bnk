package pin

import (
	"crypto/tls"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGenerateCertProducesUsableTLSCertAndStableFingerprint(t *testing.T) {
	certPEM, keyPEM, err := GenerateCert()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tls.X509KeyPair(certPEM, keyPEM); err != nil {
		t.Fatalf("generated cert unusable: %v", err)
	}
	fp1, err := Fingerprint(certPEM)
	if err != nil {
		t.Fatal(err)
	}
	fp2, err := Fingerprint(certPEM)
	if err != nil {
		t.Fatal(err)
	}
	if fp1 != fp2 {
		t.Errorf("fingerprint not stable: %s vs %s", fp1, fp2)
	}
	if len(fp1) != 64 || strings.ToLower(fp1) != fp1 {
		t.Errorf("fingerprint = %q, want 64 lowercase hex chars", fp1)
	}
}

// pinnedServer starts a TLS server with a generated cert and returns its
// URL and fingerprint.
func pinnedServer(t *testing.T) (*httptest.Server, string) {
	t.Helper()
	certPEM, keyPEM, err := GenerateCert()
	if err != nil {
		t.Fatal(err)
	}
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	fp, err := Fingerprint(certPEM)
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "pinned")
	}))
	ts.TLS = &tls.Config{Certificates: []tls.Certificate{cert}}
	ts.StartTLS()
	t.Cleanup(ts.Close)
	return ts, fp
}

func TestClientTLSConfigAcceptsPinnedCert(t *testing.T) {
	ts, fp := pinnedServer(t)
	hc := &http.Client{Transport: &http.Transport{TLSClientConfig: ClientTLSConfig(fp)}}
	resp, err := hc.Get(ts.URL)
	if err != nil {
		t.Fatalf("pinned client rejected matching cert: %v", err)
	}
	resp.Body.Close()
}

func TestClientTLSConfigRejectsWrongFingerprint(t *testing.T) {
	ts, _ := pinnedServer(t)
	wrong := strings.Repeat("ab", 32)
	hc := &http.Client{Transport: &http.Transport{TLSClientConfig: ClientTLSConfig(wrong)}}
	resp, err := hc.Get(ts.URL)
	if err == nil {
		resp.Body.Close()
		t.Fatal("client with wrong fingerprint connected, want failure")
	}
}

func TestEnrollKeyFormatRoundTrip(t *testing.T) {
	fp := strings.Repeat("cd", 32)
	full := FormatEnrollKey("deadbeef", fp)
	if !strings.HasPrefix(full, "bnkkey:") {
		t.Errorf("formatted key = %q, want bnkkey: prefix", full)
	}
	secret, gotFP, err := ParseEnrollKey(full)
	if err != nil {
		t.Fatal(err)
	}
	if secret != "deadbeef" || gotFP != fp {
		t.Errorf("parsed = %q %q, want deadbeef %q", secret, gotFP, fp)
	}
}

func TestParseEnrollKeyRejectsGarbage(t *testing.T) {
	for _, s := range []string{"", "bnkkey:", "nope", "bnkkey:onlysecret"} {
		if _, _, err := ParseEnrollKey(s); err == nil {
			t.Errorf("ParseEnrollKey(%q) succeeded, want error", s)
		}
	}
}
