package server

import (
	"encoding/json"
	"net/http"
	"net/netip"

	"vpnmesh/internal/acl"
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
	mux.HandleFunc("PUT /policy", func(w http.ResponseWriter, r *http.Request) {
		var p acl.Policy
		if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := s.SetPolicy(&p); err != nil {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /policy", func(w http.ResponseWriter, r *http.Request) {
		p := s.Policy()
		if p == nil {
			http.Error(w, "no policy set (allow all)", http.StatusNotFound)
			return
		}
		json.NewEncoder(w).Encode(p)
	})
	mux.HandleFunc("GET /check", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		allowed, err := s.CheckPolicy(q.Get("src"), q.Get("dst"), q.Get("target"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]bool{"allowed": allowed})
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
