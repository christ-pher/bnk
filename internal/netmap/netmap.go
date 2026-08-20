// Package netmap defines the data model the control server pushes to
// clients: who your peers are, their keys, addresses, and endpoints.
package netmap

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/netip"
)

type NodeID uint32

// Key is a 32-byte public key (WireGuard node key or disco key). It
// marshals to JSON as base64 so state files and wire messages stay readable.
type Key [32]byte

func (k Key) String() string {
	return base64.StdEncoding.EncodeToString(k[:])
}

func (k Key) MarshalJSON() ([]byte, error) {
	return json.Marshal(k.String())
}

func (k *Key) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return fmt.Errorf("key %q: %w", s, err)
	}
	if len(raw) != len(k) {
		return fmt.Errorf("key %q: got %d bytes, want %d", s, len(raw), len(k))
	}
	copy(k[:], raw)
	return nil
}

type Netmap struct {
	SelfID NodeID       `json:"self_id"`
	SelfIP netip.Prefix `json:"self_ip"`
	Peers  []Peer       `json:"peers"`
}

type Peer struct {
	ID        NodeID           `json:"id"`
	Name      string           `json:"name"`
	NodeKey   Key              `json:"node_key"`
	DiscoKey  Key              `json:"disco_key"`
	IP        netip.Addr       `json:"ip"`
	Endpoints []netip.AddrPort `json:"endpoints,omitempty"`
	Online    bool             `json:"online"`
	Tags      []string         `json:"tags,omitempty"`
}
