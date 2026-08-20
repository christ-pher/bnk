package server

import (
	"encoding/json"
	"net/http"
	"net/netip"

	"vpnmesh/internal/netmap"
	"vpnmesh/internal/pin"
)

// AdminNode is the CLI-facing view of a node.
type AdminNode struct {
	ID     netmap.NodeID `json:"id"`
	Name   string        `json:"name"`
	OS     string        `json:"os,omitempty"`
	IP     netip.Addr    `json:"ip"`
	Online bool          `json:"online"`
	Tags   []string      `json:"tags,omitempty"`
}

// AdminHandler serves the local admin API (bound to a unix socket by vpnd).
// fingerprint is the server cert's fingerprint, embedded in minted
// enrollment keys.
func (s *Server) AdminHandler(fingerprint string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /nodes", func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		nodes := make([]AdminNode, 0, len(s.st.Nodes))
		for _, n := range s.st.Nodes {
			_, online := s.sessions[n.ID]
			nodes = append(nodes, AdminNode{
				ID: n.ID, Name: n.Name, OS: n.OS, IP: n.IP, Online: online, Tags: n.Tags,
			})
		}
		s.mu.Unlock()
		json.NewEncoder(w).Encode(nodes)
	})
	mux.HandleFunc("POST /enroll-keys", func(w http.ResponseWriter, r *http.Request) {
		secret, err := s.NewEnrollKey()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"key": pin.FormatEnrollKey(secret, fingerprint)})
	})
	return mux
}
