package server

import (
	"fmt"
	"net/netip"

	"github.com/christ-pher/bnk/internal/ipam"
)

// Network reports the prefix the mesh allocates addresses from.
func (s *Server) Network() netip.Prefix {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st.Prefix
}

// SetNetwork changes the mesh prefix and re-addresses every enrolled
// node, keeping host numbers where they still fit. Connected clients
// pick the change up from the netmap push and re-address themselves; a
// node that is offline gets it when it reconnects.
//
// Nothing is persisted unless the whole reassignment succeeds, so a
// rejected prefix leaves the mesh exactly as it was.
func (s *Server) SetNetwork(p netip.Prefix) error {
	if !p.Addr().Is4() {
		return fmt.Errorf("mesh network must be IPv4, got %v", p)
	}
	// A /31 or /32 has no assignable host addresses, so it would be
	// accepted on an empty mesh and then reject the first node to join.
	if p.Bits() > 30 {
		return fmt.Errorf("mesh network %v is too small: use /30 or larger", p)
	}
	p = p.Masked()

	s.mu.Lock()
	old := s.st.Prefix
	if p == old {
		s.mu.Unlock()
		return nil
	}

	current := make([]netip.Addr, len(s.st.Nodes))
	for i, n := range s.st.Nodes {
		current[i] = n.IP
	}
	mapping, err := ipam.Reassign(old, p, current)
	if err != nil {
		s.mu.Unlock()
		return err
	}

	for i := range s.st.Nodes {
		s.st.Nodes[i].IP = mapping[s.st.Nodes[i].IP]
	}
	s.st.Prefix = p

	if err := s.fs.Save(s.st); err != nil {
		// Roll back in memory (current still holds the old addresses) so
		// a failed save cannot leave the server handing out addresses it
		// never persisted.
		for i := range s.st.Nodes {
			s.st.Nodes[i].IP = current[i]
		}
		s.st.Prefix = old
		s.mu.Unlock()
		return err
	}
	s.mu.Unlock()

	s.broadcastNetmaps()
	return nil
}
