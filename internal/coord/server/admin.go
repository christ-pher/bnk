package server

import (
	"encoding/json"
	"net/http"
	"net/netip"
	"time"

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

// AdminKey is the CLI-facing view of an enrollment key: the secret itself
// is only revealed at mint time.
type AdminKey struct {
	Prefix    string    `json:"prefix"`
	Reusable  bool      `json:"reusable"`
	Used      bool      `json:"used"`
	Revoked   bool      `json:"revoked"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
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
		ttl := 24 * time.Hour
		if v := r.URL.Query().Get("ttl"); v != "" {
			d, err := time.ParseDuration(v)
			if err != nil {
				http.Error(w, "bad ttl: "+err.Error(), http.StatusBadRequest)
				return
			}
			ttl = d
		}
		reusable := r.URL.Query().Get("reusable") == "true"
		secret, err := s.MintEnrollKey(ttl, reusable)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"key": pin.FormatEnrollKey(secret, fingerprint)})
	})
	mux.HandleFunc("GET /enroll-keys", func(w http.ResponseWriter, r *http.Request) {
		var out []AdminKey
		for _, k := range s.EnrollKeys() {
			out = append(out, AdminKey{
				Prefix: k.Secret[:8], Reusable: k.Reusable, Used: k.Used,
				Revoked: k.Revoked, CreatedAt: k.CreatedAt, ExpiresAt: k.ExpiresAt,
			})
		}
		json.NewEncoder(w).Encode(out)
	})
	mux.HandleFunc("POST /enroll-keys/revoke", func(w http.ResponseWriter, r *http.Request) {
		if err := s.RevokeEnrollKey(r.URL.Query().Get("prefix")); err != nil {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	return mux
}
