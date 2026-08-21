package selfupdate

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// assetServer serves one named release asset and a SHA256SUMS that
// carries the given digest line for it.
func assetServer(t *testing.T, tag, asset string, body []byte, sums string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/o/r/releases/download/"+tag+"/"+asset, func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	})
	mux.HandleFunc("/o/r/releases/download/"+tag+"/SHA256SUMS", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, sums)
	})
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

func TestFetchVerifiedReturnsTheAssetWhenTheChecksumMatches(t *testing.T) {
	msi := []byte("not really an msi")
	sums := fmt.Sprintf("%x  bnk-windows-amd64.msi\n", sha256.Sum256(msi))
	ts := assetServer(t, "v1.2.3", "bnk-windows-amd64.msi", msi, sums)

	got, err := FetchVerified(ts.URL+"/o/r", "v1.2.3", "bnk-windows-amd64.msi")
	if err != nil {
		t.Fatalf("FetchVerified: %v", err)
	}
	if string(got) != string(msi) {
		t.Errorf("got %q, want the asset bytes", got)
	}
}

// A served asset that does not match SHA256SUMS must never be returned:
// the caller is about to execute it.
func TestFetchVerifiedRejectsATamperedAsset(t *testing.T) {
	genuine := []byte("the release that was checksummed")
	sums := fmt.Sprintf("%x  bnk-windows-amd64.msi\n", sha256.Sum256(genuine))
	ts := assetServer(t, "v1.2.3", "bnk-windows-amd64.msi", []byte("swapped"), sums)

	if _, err := FetchVerified(ts.URL+"/o/r", "v1.2.3", "bnk-windows-amd64.msi"); err == nil {
		t.Fatal("a tampered asset was returned as verified")
	}
}

// An asset SHA256SUMS does not list cannot be verified, so it must not
// be returned either — absence of a checksum is not a pass.
func TestFetchVerifiedRejectsAnUnlistedAsset(t *testing.T) {
	msi := []byte("present but never checksummed")
	ts := assetServer(t, "v1.2.3", "bnk-windows-amd64.msi", msi, "0000  something-else\n")

	if _, err := FetchVerified(ts.URL+"/o/r", "v1.2.3", "bnk-windows-amd64.msi"); err == nil {
		t.Fatal("an unlisted asset was returned as verified")
	}
}
