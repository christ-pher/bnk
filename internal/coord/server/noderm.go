package server

import (
	"fmt"

	"github.com/christ-pher/bnk/internal/netmap"
)

// RemoveNode forgets a node by name, freeing its address for reuse. The
// node's peers learn about it from the netmap push; the node itself, if
// still running, is rejected the next time it reconnects and has to
// re-enroll with a fresh key.
func (s *Server) RemoveNode(name string) error {
	s.mu.Lock()
	idx := -1
	for i, n := range s.st.Nodes {
		if n.Name == name {
			idx = i
			break
		}
	}
	if idx < 0 {
		s.mu.Unlock()
		return fmt.Errorf("no node named %q", name)
	}
	return s.removeAtLocked(idx)
}

// removeAtLocked removes the node at idx and releases s.mu.
func (s *Server) removeAtLocked(idx int) error {
	removed := s.st.Nodes[idx]
	s.st.Nodes = append(s.st.Nodes[:idx], s.st.Nodes[idx+1:]...)
	if err := s.fs.Save(s.st); err != nil {
		// Put it back: the mesh must match what is on disk.
		s.st.Nodes = append(s.st.Nodes, removed)
		s.mu.Unlock()
		return err
	}
	sess := s.sessions[removed.ID]
	delete(s.sessions, removed.ID)
	delete(s.endpoints, removed.ID)
	s.mu.Unlock()

	// Drop the node's own session so it stops receiving netmaps at once.
	if sess != nil {
		sess.conn.Close()
	}
	s.broadcastNetmaps()
	return nil
}

// removeNodeByID removes the node an authenticated session belongs to.
// It matches on ID rather than resolving to a name first: names are
// chosen by the enrolling node, so a name lookup could delete a
// different node than the one that asked to leave.
func (s *Server) removeNodeByID(id netmap.NodeID) error {
	s.mu.Lock()
	idx := -1
	for i, n := range s.st.Nodes {
		if n.ID == id {
			idx = i
			break
		}
	}
	if idx < 0 {
		s.mu.Unlock()
		return fmt.Errorf("no node with id %d", id)
	}
	return s.removeAtLocked(idx)
}
