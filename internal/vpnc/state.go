package vpnc

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"io/fs"
	"net/netip"
	"os"
	"path/filepath"

	"golang.org/x/crypto/curve25519"

	"vpnmesh/internal/netmap"
)

// state is the client's persistent identity and enrollment result.
type state struct {
	PrivateKey  netmap.Key    `json:"private_key"`
	DiscoPriv   netmap.Key    `json:"disco_priv,omitempty"`
	DiscoPub    netmap.Key    `json:"disco_pub,omitempty"`
	ServerURL   string        `json:"server_url"`
	Fingerprint string        `json:"fingerprint,omitempty"`
	NodeID      netmap.NodeID `json:"node_id,omitempty"`
	IP          netip.Addr    `json:"ip,omitempty"`
	Prefix      netip.Prefix  `json:"prefix,omitempty"`
}

func statePath(dir string) string {
	return filepath.Join(dir, "client.json")
}

func loadState(dir string) (state, bool, error) {
	raw, err := os.ReadFile(statePath(dir))
	if errors.Is(err, fs.ErrNotExist) {
		return state{}, false, nil
	}
	if err != nil {
		return state{}, false, err
	}
	var st state
	if err := json.Unmarshal(raw, &st); err != nil {
		return state{}, false, err
	}
	return st, true, nil
}

func saveState(dir string, st state) error {
	raw, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".client-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), statePath(dir))
}

func generateKeypair() (priv, pub netmap.Key, err error) {
	if _, err = rand.Read(priv[:]); err != nil {
		return
	}
	priv[0] &= 248
	priv[31] &= 127
	priv[31] |= 64
	p, err := curve25519.X25519(priv[:], curve25519.Basepoint)
	if err != nil {
		return
	}
	copy(pub[:], p)
	return priv, pub, nil
}

func publicKey(priv netmap.Key) (netmap.Key, error) {
	p, err := curve25519.X25519(priv[:], curve25519.Basepoint)
	if err != nil {
		return netmap.Key{}, err
	}
	var pub netmap.Key
	copy(pub[:], p)
	return pub, nil
}
