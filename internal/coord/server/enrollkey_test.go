package server_test

import (
	"testing"
	"time"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/coord/client"
	"vpnmesh/internal/netmap"
)

func enrollWith(t *testing.T, e *env, secret, name string, key netmap.Key) error {
	t.Helper()
	_, err := client.Enroll(t.Context(), e.ts.URL, e.ts.Client(), coord.EnrollRequest{
		EnrollKey: secret, Hostname: name, NodeKey: key,
	})
	return err
}

func TestOneTimeKeyAdmitsExactlyOneNode(t *testing.T) {
	e := startServer(t)
	secret, err := e.srv.MintEnrollKey(time.Hour, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := enrollWith(t, e, secret, "alpha", key32(1)); err != nil {
		t.Fatalf("first enroll: %v", err)
	}
	if err := enrollWith(t, e, secret, "mallory", key32(2)); err == nil {
		t.Fatal("one-time key admitted a second node")
	}
}

func TestReusableKeyAdmitsSeveral(t *testing.T) {
	e := startServer(t)
	secret, err := e.srv.MintEnrollKey(time.Hour, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, name := range []string{"a", "b", "c"} {
		if err := enrollWith(t, e, secret, name, key32(byte(10+i))); err != nil {
			t.Fatalf("enroll %s: %v", name, err)
		}
	}
}

func TestExpiredKeyIsRejected(t *testing.T) {
	e := startServer(t)
	secret, err := e.srv.MintEnrollKey(-time.Second, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := enrollWith(t, e, secret, "late", key32(3)); err == nil {
		t.Fatal("expired key admitted a node")
	}
}

func TestKeyListAndRevoke(t *testing.T) {
	e := startServer(t)
	secret, err := e.srv.MintEnrollKey(time.Hour, true)
	if err != nil {
		t.Fatal(err)
	}

	keys := e.srv.EnrollKeys()
	found := false
	for _, k := range keys {
		if k.Secret == secret && k.Reusable && !k.Used && !k.Revoked {
			found = true
		}
	}
	if !found {
		t.Fatalf("minted key not listed correctly: %+v", keys)
	}

	if err := e.srv.RevokeEnrollKey(secret[:8]); err != nil {
		t.Fatal(err)
	}
	if err := enrollWith(t, e, secret, "late", key32(4)); err == nil {
		t.Fatal("revoked key admitted a node")
	}
	if err := e.srv.RevokeEnrollKey("nope1234"); err == nil {
		t.Fatal("revoking an unknown prefix succeeded")
	}
}
