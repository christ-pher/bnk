package vpnc

import (
	"net/netip"
	"time"

	"github.com/christ-pher/bnk/internal/netmap"
)

// Status is served on the local unix socket for the bnk CLI.
type Status struct {
	// Running reports whether the tunnel is up. False after `bnk down`:
	// the daemon still answers, but there is no interface or session.
	Running bool `json:"running"`
	// Enrolled is false on a machine that has been installed but never
	// given a key, which is a different problem from being disconnected
	// and needs a different offer from the UI.
	Enrolled bool         `json:"enrolled"`
	Self     SelfStatus   `json:"self"`
	Peers    []PeerStatus `json:"peers"`
}

type SelfStatus struct {
	ID   netmap.NodeID `json:"id"`
	Name string        `json:"name"`
	IP   netip.Addr    `json:"ip"`
}

type PeerStatus struct {
	ID            netmap.NodeID `json:"id"`
	Name          string        `json:"name"`
	IP            netip.Addr    `json:"ip"`
	Online        bool          `json:"online"`
	Path          string        `json:"path"` // "direct" or "relay"
	LastHandshake time.Time     `json:"last_handshake,omitempty"`
}
