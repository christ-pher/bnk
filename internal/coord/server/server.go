// Package server implements the control server: enrollment, session
// registry, and netmap computation/push.
package server

import (
	"bufio"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"strconv"
	"strings"
	"sync"
	"time"

	"vpnmesh/internal/acl"
	"vpnmesh/internal/coord"
	"vpnmesh/internal/ipam"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/store"
)

var defaultPrefix = netip.MustParsePrefix("100.64.0.0/10")

// Liveness tuning: the server sends keepalives every KeepaliveInterval
// and drops sessions silent for ReadTimeout. Package-level for tests.
var (
	KeepaliveInterval = 25 * time.Second
	ReadTimeout       = 75 * time.Second
)

type Server struct {
	fs          *store.FileStore
	readTimeout time.Duration // captured from ReadTimeout at New
	keepalive   time.Duration // captured from KeepaliveInterval at New

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
	done chan struct{}
}

// keepaliveLoop proves liveness to the client (whose read deadline would
// otherwise fire on a quiet mesh).
func (s *session) keepaliveLoop(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			if err := s.send(coord.FrameKeepalive, nil); err != nil {
				return
			}
		}
	}
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
		fs:          fs,
		readTimeout: ReadTimeout,
		keepalive:   KeepaliveInterval,
		st:          st,
		sessions:    make(map[netmap.NodeID]*session),
		endpoints:   make(map[netmap.NodeID][]netip.AddrPort),
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
	sess := &session{id: node.ID, conn: conn, bw: bufrw.Writer, done: make(chan struct{})}
	if old, exists := s.sessions[node.ID]; exists {
		old.conn.Close()
	}
	s.sessions[node.ID] = sess
	s.mu.Unlock()

	s.broadcastNetmaps()
	go sess.keepaliveLoop(s.keepalive)
	s.readLoop(sess, bufrw.Reader)
}

func (s *Server) readLoop(sess *session, r *bufio.Reader) {
	defer func() {
		close(sess.done)
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
		// Clients keepalive more often than ReadTimeout; total silence
		// means the machine or its network path is gone.
		sess.conn.SetReadDeadline(time.Now().Add(s.readTimeout))
		typ, payload, err := coord.ReadFrame(r)
		if err != nil {
			return
		}
		switch typ {
		case coord.FrameKeepalive:
		case coord.FrameRelayData:
			dst, pkt, err := coord.DecodeRelay(payload)
			if err != nil {
				return
			}
			s.mu.Lock()
			target, online := s.sessions[dst]
			s.mu.Unlock()
			if !online {
				continue // peer offline: drop silently, sender's WG will retry
			}
			// Restamp with the true source so identity can't be spoofed.
			if err := target.send(coord.FrameRelayData, coord.EncodeRelay(sess.id, pkt)); err != nil {
				target.conn.Close()
			}
		case coord.FrameControl:
			msg, err := coord.DecodeControl(payload)
			if err != nil {
				return
			}
			switch msg.T {
			case coord.MsgEndpoints:
				s.mu.Lock()
				s.endpoints[sess.id] = msg.Endpoints
				s.mu.Unlock()
				s.broadcastNetmaps()
			case coord.MsgDiscoFwd:
				if msg.DiscoFwd == nil {
					continue
				}
				s.mu.Lock()
				target, online := s.sessions[msg.DiscoFwd.Dst]
				s.mu.Unlock()
				if !online {
					continue
				}
				// Restamp with the true source; the sender's Src is ignored.
				out, err := coord.EncodeControl(coord.Envelope{T: coord.MsgDiscoFwd, DiscoFwd: &coord.DiscoFwd{
					Src: sess.id, Payload: msg.DiscoFwd.Payload,
				}})
				if err != nil {
					continue
				}
				if err := target.send(coord.FrameControl, out); err != nil {
					target.conn.Close()
				}
			}
		}
	}
}

// SetPolicy validates, persists, and broadcasts a new ACL policy; nil
// clears it (allow-all).
func (s *Server) SetPolicy(p *acl.Policy) error {
	s.mu.Lock()
	if p != nil {
		if _, err := acl.Compile(*p, s.aclNodesLocked()); err != nil {
			s.mu.Unlock()
			return err
		}
	}
	s.st.Policy = p
	err := s.fs.Save(s.st)
	s.mu.Unlock()
	if err != nil {
		return err
	}
	s.broadcastNetmaps()
	return nil
}

// CheckPolicy is a dry-run evaluator: would traffic from node src to node
// dst matching target ("tcp/22", "icmp") be allowed under the current
// policy? Return-flow admission is not modeled — this answers "could src
// initiate to dst".
func (s *Server) CheckPolicy(src, dst, target string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var srcNode *store.Node
	for i, n := range s.st.Nodes {
		if n.Name == src {
			srcNode = &s.st.Nodes[i]
		}
	}
	if srcNode == nil {
		return false, fmt.Errorf("unknown source node %q", src)
	}
	proto, portSpec, _ := strings.Cut(target, "/")
	var port uint64
	if proto != "icmp" {
		var err error
		port, err = strconv.ParseUint(portSpec, 10, 16)
		if err != nil {
			return false, fmt.Errorf("bad target %q (want tcp/22, udp/53, or icmp)", target)
		}
	}
	compiled, enabled := s.compileFilterLocked()
	if !enabled {
		return true, nil
	}
	for _, r := range compiled[dst] {
		if r.Match(srcNode.IP, proto, uint16(port)) {
			return true, nil
		}
	}
	return false, nil
}

// Policy returns the current ACL policy (nil = allow all).
func (s *Server) Policy() *acl.Policy {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st.Policy
}

func (s *Server) aclNodesLocked() []acl.NodeInfo {
	nodes := make([]acl.NodeInfo, 0, len(s.st.Nodes))
	for _, n := range s.st.Nodes {
		nodes = append(nodes, acl.NodeInfo{Name: n.Name, IP: n.IP, Tags: n.Tags})
	}
	return nodes
}

// compileFilterLocked compiles the current policy; s.mu must be held. A nil
// map with enabled=false means no policy (allow all). Compile errors after
// a node change fail closed: enforcement stays on with empty rules.
func (s *Server) compileFilterLocked() (map[string][]acl.CompiledRule, bool) {
	if s.st.Policy == nil {
		return nil, false
	}
	compiled, err := acl.Compile(*s.st.Policy, s.aclNodesLocked())
	if err != nil {
		return map[string][]acl.CompiledRule{}, true
	}
	return compiled, true
}

// netmapForLocked computes the netmap for one node; s.mu must be held.
func (s *Server) netmapForLocked(node store.Node) netmap.Netmap {
	nm := netmap.Netmap{
		SelfID: node.ID,
		SelfIP: netip.PrefixFrom(node.IP, 32),
	}
	compiled, enabled := s.compileFilterLocked()
	nm.FilterEnabled = enabled
	nm.Filter = compiled[node.Name]
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
