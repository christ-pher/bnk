package pin_test

import (
	"crypto/tls"
	"encoding/pem"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/christ-pher/bnk/internal/pin"
)

func der(t *testing.T, certPEM []byte) []byte {
	t.Helper()
	block, _ := pem.Decode(certPEM)
	if block == nil {
		t.Fatal("no PEM block")
	}
	return block.Bytes
}

// A pinned client must reject a server that presents someone else's
// certificate alongside its own. Only the leaf is bound to the handshake
// signature, so honouring a fingerprint match deeper in the chain lets
// any attacker append the real (public) certificate and impersonate the
// server outright.
func TestPinRejectsRealCertAppendedToAttackerChain(t *testing.T) {
	realCert, _, err := pin.GenerateCert()
	if err != nil {
		t.Fatal(err)
	}
	realFP, err := pin.Fingerprint(realCert)
	if err != nil {
		t.Fatal(err)
	}

	// The attacker holds only its own key, and appends the victim's
	// public certificate — which anyone can fetch by connecting.
	attackerCert, attackerKey, err := pin.GenerateCert()
	if err != nil {
		t.Fatal(err)
	}
	pair, err := tls.X509KeyPair(attackerCert, attackerKey)
	if err != nil {
		t.Fatal(err)
	}
	pair.Certificate = [][]byte{der(t, attackerCert), der(t, realCert)}

	ln, err := tls.Listen("tcp", "127.0.0.1:0", &tls.Config{Certificates: []tls.Certificate{pair}})
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				c.(*tls.Conn).Handshake()
				time.Sleep(50 * time.Millisecond)
			}()
		}
	}()

	conn, err := net.DialTimeout("tcp", ln.Addr().String(), 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	tc := tls.Client(conn, pin.ClientTLSConfig(realFP))
	tc.SetDeadline(time.Now().Add(5 * time.Second))

	if err := tc.Handshake(); err == nil {
		t.Fatal("PIN BYPASSED: handshake succeeded against a server holding only the attacker's key, " +
			"because the real certificate was appended to the chain")
	} else if !strings.Contains(err.Error(), "pinned fingerprint") {
		t.Fatalf("rejected, but for the wrong reason: %v", err)
	}
}

// The honest case must still work.
func TestPinAcceptsTheRealServer(t *testing.T) {
	certPEM, keyPEM, err := pin.GenerateCert()
	if err != nil {
		t.Fatal(err)
	}
	fp, err := pin.Fingerprint(certPEM)
	if err != nil {
		t.Fatal(err)
	}
	pair, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatal(err)
	}
	ln, err := tls.Listen("tcp", "127.0.0.1:0", &tls.Config{Certificates: []tls.Certificate{pair}})
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		defer c.Close()
		c.(*tls.Conn).Handshake()
		time.Sleep(50 * time.Millisecond)
	}()

	conn, err := net.DialTimeout("tcp", ln.Addr().String(), 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	tc := tls.Client(conn, pin.ClientTLSConfig(fp))
	tc.SetDeadline(time.Now().Add(5 * time.Second))
	if err := tc.Handshake(); err != nil {
		t.Fatalf("pinned client rejected the genuine server: %v", err)
	}
}
