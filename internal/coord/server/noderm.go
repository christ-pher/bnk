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
	id := s.st.Nodes[idx].ID
	removed := s.st.Nodes[idx]
	s.st.Nodes = append(s.st.Nodes[:idx], s.st.Nodes[idx+1:]...)
	if err := s.fs.Save(s.st); err != nil {
		// Put it back: the mesh must match what is on disk.
		s.st.Nodes = append(s.st.Nodes, removed)
		s.mu.Unlock()
		return err
	}
	sess := s.sessions[id]
	delete(s.sessions, id)
	delete(s.endpoints, id)
	s.mu.Unlock()

	// Drop the node's own session so it stops receiving netmaps at once.
	if sess != nil {
		sess.conn.Close()
	}
	s.broadcastNetmaps()
	return nil
}

// removeNodeByID backs the session leave path, where the caller is
// identified by its authenticated session rather than by name.
func (s *Server) removeNodeByID(id netmap.NodeID) error {
	s.mu.Lock()
	name := ""
	for _, n := range s.st.Nodes {
		if n.ID == id {
			name = n.Name
			break
		}
	}
	s.mu.Unlock()
	if name == "" {
		return fmt.Errorf("no node with id %d", id)
	}
	return s.RemoveNode(name)
}
