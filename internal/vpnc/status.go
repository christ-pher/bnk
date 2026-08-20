package vpnc

import (
	"net/netip"
	"time"

	"vpnmesh/internal/netmap"
)

// Status is served on the local unix socket for the vpn CLI.
type Status struct {
	Self  SelfStatus   `json:"self"`
	Peers []PeerStatus `json:"peers"`
}

type SelfStatus struct {
	ID netmap.NodeID `json:"id"`
	IP netip.Addr    `json:"ip"`
}

type PeerStatus struct {
	ID            netmap.NodeID `json:"id"`
	Name          string        `json:"name"`
	IP            netip.Addr    `json:"ip"`
	Online        bool          `json:"online"`
	Path          string        `json:"path"` // "direct" or "relay"
	LastHandshake time.Time     `json:"last_handshake,omitempty"`
}
