// Package server implements the control server: enrollment, session
// registry, and netmap computation/push.
package server

import (
	"bufio"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"net/netip"
	"sync"
	"time"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/ipam"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/store"
)

var defaultPrefix = netip.MustParsePrefix("100.64.0.0/10")

type Server struct {
	fs *store.FileStore

	mu        sync.Mutex
	st        store.State
	sessions  map[netmap.NodeID]*session
	endpoints map[netmap.NodeID][]netip.AddrPort
}

// session is one connected client; writes are serialized by wmu.
type session struct {
	id   netmap.NodeID
	conn net.Conn
	wmu  sync.Mutex
	bw   *bufio.Writer
}

func (s *session) send(typ coord.FrameType, payload []byte) error {
	s.wmu.Lock()
	defer s.wmu.Unlock()
	if err := coord.WriteFrame(s.bw, typ, payload); err != nil {
		return err
	}
	return s.bw.Flush()
}

func New(fs *store.FileStore) (*Server, error) {
	st, err := fs.Load()
	if err != nil {
		return nil, err
	}
	if !st.Prefix.IsValid() {
		st.Prefix = defaultPrefix
	}
	return &Server{
		fs:        fs,
		st:        st,
		sessions:  make(map[netmap.NodeID]*session),
		endpoints: make(map[netmap.NodeID][]netip.AddrPort),
	}, nil
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /enroll", s.handleEnroll)
	mux.HandleFunc("GET /c", s.handleSession)
	return mux
}

// NewEnrollKey mints and persists a new enrollment secret.
func (s *Server) NewEnrollKey() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	secret := hex.EncodeToString(raw)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.EnrollKeys = append(s.st.EnrollKeys, store.EnrollKey{Secret: secret, CreatedAt: time.Now().UTC()})
	if err := s.fs.Save(s.st); err != nil {
		return "", err
	}
	return secret, nil
}

func (s *Server) validEnrollKey(secret string) bool {
	for _, k := range s.st.EnrollKeys {
		if !k.Revoked && subtle.ConstantTimeCompare([]byte(k.Secret), []byte(secret)) == 1 {
			return true
		}
	}
	return false
}

func (s *Server) handleEnroll(w http.ResponseWriter, r *http.Request) {
	var req coord.EnrollRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	if !s.validEnrollKey(req.EnrollKey) {
		s.mu.Unlock()
		http.Error(w, "invalid enrollment key", http.StatusForbidden)
		return
	}

	node, ok := s.nodeByKey(req.NodeKey)
	if !ok {
		used := make(map[netip.Addr]bool)
		var maxID netmap.NodeID
		for _, n := range s.st.Nodes {
			used[n.IP] = true
			if n.ID > maxID {
				maxID = n.ID
			}
		}
		ip, err := ipam.Next(s.st.Prefix, used)
		if err != nil {
			s.mu.Unlock()
			http.Error(w, err.Error(), http.StatusInsufficientStorage)
			return
		}
		node = store.Node{
			ID:        maxID + 1,
			Name:      req.Hostname,
			OS:        req.OS,
			NodeKey:   req.NodeKey,
			DiscoKey:  req.DiscoKey,
			IP:        ip,
			CreatedAt: time.Now().UTC(),
		}
		s.st.Nodes = append(s.st.Nodes, node)
		if err := s.fs.Save(s.st); err != nil {
			s.mu.Unlock()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	prefix := s.st.Prefix
	s.mu.Unlock()

	s.broadcastNetmaps()
	json.NewEncoder(w).Encode(coord.EnrollResponse{NodeID: node.ID, IP: node.IP, Prefix: prefix})
}

func (s *Server) nodeByKey(key netmap.Key) (store.Node, bool) {
	for _, n := range s.st.Nodes {
		if n.NodeKey == key {
			return n, true
		}
	}
	return store.Node{}, false
}

func (s *Server) handleSession(w http.ResponseWriter, r *http.Request) {
	hj, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "hijack unsupported", http.StatusInternalServerError)
		return
	}
	conn, bufrw, err := hj.Hijack()
	if err != nil {
		return
	}
	if _, err := conn.Write([]byte("HTTP/1.1 101 Switching Protocols\r\nUpgrade: vpn-coord/1\r\nConnection: Upgrade\r\n\r\n")); err != nil {
		conn.Close()
		return
	}

	// First frame must be a hello identifying an enrolled node.
	typ, payload, err := coord.ReadFrame(bufrw.Reader)
	if err != nil || typ != coord.FrameControl {
		conn.Close()
		return
	}
	msg, err := coord.DecodeControl(payload)
	if err != nil || msg.T != coord.MsgHello || msg.Hello == nil {
		conn.Close()
		return
	}

	s.mu.Lock()
	node, ok := s.nodeByKey(msg.Hello.NodeKey)
	if !ok {
		s.mu.Unlock()
		conn.Close()
		return
	}
	sess := &session{id: node.ID, conn: conn, bw: bufrw.Writer}
	if old, exists := s.sessions[node.ID]; exists {
		old.conn.Close()
	}
	s.sessions[node.ID] = sess
	s.mu.Unlock()

	s.broadcastNetmaps()
	s.readLoop(sess, bufrw.Reader)
}

func (s *Server) readLoop(sess *session, r *bufio.Reader) {
	defer func() {
		s.mu.Lock()
		if s.sessions[sess.id] == sess {
			delete(s.sessions, sess.id)
			delete(s.endpoints, sess.id)
		}
		s.mu.Unlock()
		sess.conn.Close()
		s.broadcastNetmaps()
	}()
	for {
		typ, payload, err := coord.ReadFrame(r)
		if err != nil {
			return
		}
		switch typ {
		case coord.FrameKeepalive:
		case coord.FrameControl:
			msg, err := coord.DecodeControl(payload)
			if err != nil {
				return
			}
			if msg.T == coord.MsgEndpoints {
				s.mu.Lock()
				s.endpoints[sess.id] = msg.Endpoints
				s.mu.Unlock()
				s.broadcastNetmaps()
			}
		}
	}
}

// netmapForLocked computes the netmap for one node; s.mu must be held.
func (s *Server) netmapForLocked(node store.Node) netmap.Netmap {
	nm := netmap.Netmap{
		SelfID: node.ID,
		SelfIP: netip.PrefixFrom(node.IP, 32),
	}
	for _, n := range s.st.Nodes {
		if n.ID == node.ID {
			continue
		}
		_, online := s.sessions[n.ID]
		nm.Peers = append(nm.Peers, netmap.Peer{
			ID:        n.ID,
			Name:      n.Name,
			NodeKey:   n.NodeKey,
			DiscoKey:  n.DiscoKey,
			IP:        n.IP,
			Endpoints: s.endpoints[n.ID],
			Online:    online,
			Tags:      n.Tags,
		})
	}
	return nm
}

// broadcastNetmaps pushes a fresh netmap to every connected session.
func (s *Server) broadcastNetmaps() {
	s.mu.Lock()
	type push struct {
		sess    *session
		payload []byte
	}
	var pushes []push
	for _, n := range s.st.Nodes {
		sess, ok := s.sessions[n.ID]
		if !ok {
			continue
		}
		nm := s.netmapForLocked(n)
		payload, err := coord.EncodeControl(coord.Envelope{T: coord.MsgNetmap, Netmap: &nm})
		if err != nil {
			continue
		}
		pushes = append(pushes, push{sess, payload})
	}
	s.mu.Unlock()

	for _, p := range pushes {
		if err := p.sess.send(coord.FrameControl, p.payload); err != nil {
			p.sess.conn.Close()
		}
	}
}
