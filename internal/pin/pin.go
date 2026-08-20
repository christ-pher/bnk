// Package pin implements the trust bootstrap: a self-signed server cert
// whose SHA-256 fingerprint travels inside the enrollment key, then is
// pinned by clients forever. No CA, no domains.
package pin

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"math/big"
	"strings"
	"time"
)

func GenerateCert() (certPEM, keyPEM []byte, err error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, nil, err
	}
	tmpl := x509.Certificate{
		SerialNumber: serial,
		Subject:      pkix.Name{CommonName: "bnk control server"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().AddDate(10, 0, 0),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &key.PublicKey, key)
	if err != nil {
		return nil, nil, err
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return nil, nil, err
	}
	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
	return certPEM, keyPEM, nil
}

// Fingerprint returns the lowercase hex SHA-256 of the certificate's DER.
func Fingerprint(certPEM []byte) (string, error) {
	block, _ := pem.Decode(certPEM)
	if block == nil || block.Type != "CERTIFICATE" {
		return "", fmt.Errorf("pin: no certificate PEM block found")
	}
	sum := sha256.Sum256(block.Bytes)
	return hex.EncodeToString(sum[:]), nil
}

// ClientTLSConfig trusts exactly one certificate: the one whose DER hashes
// to fingerprint. Chain and hostname verification are replaced, not merely
// skipped.
func ClientTLSConfig(fingerprint string) *tls.Config {
	want := strings.ToLower(fingerprint)
	return &tls.Config{
		InsecureSkipVerify: true,
		VerifyPeerCertificate: func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
			for _, raw := range rawCerts {
				sum := sha256.Sum256(raw)
				if hex.EncodeToString(sum[:]) == want {
					return nil
				}
			}
			return fmt.Errorf("pin: server certificate does not match pinned fingerprint")
		},
	}
}

func FormatEnrollKey(secret, fingerprint string) string {
	return fmt.Sprintf("bnkkey:%s:%s", secret, fingerprint)
}

func ParseEnrollKey(s string) (secret, fingerprint string, err error) {
	parts := strings.Split(s, ":")
	if len(parts) != 3 || parts[0] != "bnkkey" || parts[1] == "" || parts[2] == "" {
		return "", "", fmt.Errorf("pin: enrollment key must look like bnkkey:<secret>:<fingerprint>")
	}
	return parts[1], parts[2], nil
}
