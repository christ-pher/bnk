package magicsock

import (
	"crypto/rand"
	"encoding/base64"
	"testing"
)

func testNodeKey(t *testing.T) string {
	t.Helper()
	var k [32]byte
	if _, err := rand.Read(k[:]); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(k[:])
}

func TestParseEndpointReturnsIdentityEndpoint(t *testing.T) {
	b := NewBind()
	key := testNodeKey(t)

	ep, err := b.ParseEndpoint(key)
	if err != nil {
		t.Fatalf("ParseEndpoint(%q) error: %v", key, err)
	}
	if got := ep.DstToString(); got != key {
		t.Errorf("DstToString() = %q, want the node key %q", got, key)
	}
}

func TestParseEndpointRejectsNonKeys(t *testing.T) {
	b := NewBind()
	for _, s := range []string{
		"",
		"192.0.2.1:51820",
		"not-a-key",
		base64.StdEncoding.EncodeToString([]byte("short")),
	} {
		if _, err := b.ParseEndpoint(s); err == nil {
			t.Errorf("ParseEndpoint(%q) succeeded, want error", s)
		}
	}
}
