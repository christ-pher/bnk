// Package client implements the client side of the coordination protocol:
// enrollment and the long-lived framed session.
package client

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"sync"
	"time"

	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/nacl/box"

	"github.com/christ-pher/bnk/internal/coord"
	"github.com/christ-pher/bnk/internal/netmap"
)

// Enroll registers this node with the control server over plain HTTP(S).
func Enroll(ctx context.Context, baseURL string, hc *http.Client, req coord.EnrollRequest) (coord.EnrollResponse, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	body, err := json.Marshal(req)
	if err != nil {
		return coord.EnrollResponse{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/enroll", bytes.NewReader(body))
	if err != nil {
		return coord.EnrollResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := hc.Do(httpReq)
	if err != nil {
		return coord.EnrollResponse{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return coord.EnrollResponse{}, fmt.Errorf("enroll: server returned %s: %s", resp.Status, bytes.TrimSpace(msg))
	}
	var out coord.EnrollResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return coord.EnrollResponse{}, err
	}
	return out, nil
}

// Liveness tuning: a session sends keepalive frames every
// KeepaliveInterval and treats ReadTimeout of silence as a dead
// connection. Package-level so tests can shrink them.
var (
	KeepaliveInterval = 25 * time.Second
	ReadTimeout       = 75 * time.Second
)

type Handlers struct {
	OnNetmap    func(netmap.Netmap)
	OnRelayData func(src netmap.NodeID, pkt []byte)
	OnDiscoFwd  func(src netmap.NodeID, payload []byte)
}

type Session struct {
	conn        net.Conn
	wmu         sync.Mutex
	bw          *bufio.Writer
	done        chan struct{}
	keepalive   time.Duration // captured from KeepaliveInterval at Dial
	readTimeout time.Duration // captured from ReadTimeout at Dial
}

// Done is closed when the session's read loop exits for any reason.
func (s *Session) Done() <-chan struct{} {
	return s.done
}

// Dial connects to the server's /c endpoint, upgrades to the framed
// protocol, sends hello, proves possession of the node private key, and
// starts the read loop. tlsConf nil means plain TCP (tests); production
// passes a fingerprint-pinning config.
func Dial(ctx context.Context, baseURL string, tlsConf *tls.Config, nodePriv netmap.Key, h Handlers) (*Session, error) {
	pubBytes, err := curve25519.X25519(nodePriv[:], curve25519.Basepoint)
	if err != nil {
		return nil, err
	}
	var nodeKey netmap.Key
	copy(nodeKey[:], pubBytes)
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}
	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", u.Host)
	if err != nil {
		return nil, err
	}
	if tlsConf != nil {
		tc := tls.Client(conn, tlsConf)
		if err := tc.HandshakeContext(ctx); err != nil {
			conn.Close()
			return nil, err
		}
		conn = tc
	}

	br := bufio.NewReader(conn)
	req := fmt.Sprintf("GET /c HTTP/1.1\r\nHost: %s\r\nConnection: Upgrade\r\nUpgrade: bnk-coord/1\r\n\r\n", u.Host)
	if _, err := conn.Write([]byte(req)); err != nil {
		conn.Close()
		return nil, err
	}
	resp, err := http.ReadResponse(br, nil)
	if err != nil {
		conn.Close()
		return nil, err
	}
	if resp.StatusCode != http.StatusSwitchingProtocols {
		conn.Close()
		return nil, fmt.Errorf("coord dial: server returned %s, want 101", resp.Status)
	}

	s := &Session{conn: conn, bw: bufio.NewWriter(conn), done: make(chan struct{}),
		keepalive: KeepaliveInterval, readTimeout: ReadTimeout}
	hello, err := coord.EncodeControl(coord.Envelope{T: coord.MsgHello, Hello: &coord.Hello{NodeKey: nodeKey}})
	if err != nil {
		conn.Close()
		return nil, err
	}
	if err := s.send(coord.FrameControl, hello); err != nil {
		conn.Close()
		return nil, err
	}
	if err := s.answerChallenge(br, nodePriv); err != nil {
		conn.Close()
		return nil, err
	}

	go s.readLoop(br, h)
	go s.keepaliveLoop()
	return s, nil
}

// answerChallenge reads the server's challenge and returns the sealed
// proof that we hold the node private key.
func (s *Session) answerChallenge(r *bufio.Reader, nodePriv netmap.Key) error {
	s.conn.SetReadDeadline(time.Now().Add(s.readTimeout))
	typ, payload, err := coord.ReadFrame(r)
	if err != nil {
		return fmt.Errorf("coord: reading challenge: %w", err)
	}
	msg, err := coord.DecodeControl(payload)
	if err != nil || typ != coord.FrameControl || msg.T != coord.MsgChallenge || msg.Challenge == nil {
		return fmt.Errorf("coord: expected a challenge, got frame type %d", typ)
	}
	if len(msg.Challenge.Nonce) != 24 {
		return fmt.Errorf("coord: bad challenge nonce")
	}
	var nonce [24]byte
	copy(nonce[:], msg.Challenge.Nonce)
	eph := [32]byte(msg.Challenge.EphPub)
	priv := [32]byte(nodePriv)
	sealed := box.Seal(nil, msg.Challenge.Value, &nonce, &eph, &priv)
	auth, err := coord.EncodeControl(coord.Envelope{T: coord.MsgAuth, Auth: &coord.Auth{Sealed: sealed}})
	if err != nil {
		return err
	}
	return s.send(coord.FrameControl, auth)
}

// keepaliveLoop proves liveness to the server and keeps middleboxes'
// state fresh until the session dies.
func (s *Session) keepaliveLoop() {
	ticker := time.NewTicker(s.keepalive)
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

func (s *Session) readLoop(r *bufio.Reader, h Handlers) {
	defer close(s.done)
	defer s.conn.Close()
	for {
		// A connection silent past ReadTimeout is dead (the server
		// keepalives more often than this); fail it so reconnect can run.
		s.conn.SetReadDeadline(time.Now().Add(s.readTimeout))
		typ, payload, err := coord.ReadFrame(r)
		if err != nil {
			return
		}
		switch typ {
		case coord.FrameKeepalive:
		case coord.FrameRelayData:
			src, pkt, err := coord.DecodeRelay(payload)
			if err != nil {
				return
			}
			if h.OnRelayData != nil {
				h.OnRelayData(src, pkt)
			}
		case coord.FrameControl:
			msg, err := coord.DecodeControl(payload)
			if err != nil {
				return
			}
			switch {
			case msg.T == coord.MsgNetmap && msg.Netmap != nil && h.OnNetmap != nil:
				h.OnNetmap(*msg.Netmap)
			case msg.T == coord.MsgDiscoFwd && msg.DiscoFwd != nil && h.OnDiscoFwd != nil:
				h.OnDiscoFwd(msg.DiscoFwd.Src, msg.DiscoFwd.Payload)
			}
		}
	}
}

func (s *Session) send(typ coord.FrameType, payload []byte) error {
	s.wmu.Lock()
	defer s.wmu.Unlock()
	if err := coord.WriteFrame(s.bw, typ, payload); err != nil {
		return err
	}
	return s.bw.Flush()
}

// SendDiscoFwd asks the server to hand a sealed disco payload to dst —
// the bootstrap channel for hole punching when no direct path exists yet.
func (s *Session) SendDiscoFwd(dst netmap.NodeID, payload []byte) error {
	raw, err := coord.EncodeControl(coord.Envelope{T: coord.MsgDiscoFwd, DiscoFwd: &coord.DiscoFwd{Dst: dst, Payload: payload}})
	if err != nil {
		return err
	}
	return s.send(coord.FrameControl, raw)
}

// SendRelay forwards one encrypted WireGuard packet to dst via the server.
func (s *Session) SendRelay(dst netmap.NodeID, pkt []byte) error {
	return s.send(coord.FrameRelayData, coord.EncodeRelay(dst, pkt))
}

// SendEndpoints reports this node's current candidate endpoints.
func (s *Session) SendEndpoints(eps []netip.AddrPort) error {
	payload, err := coord.EncodeControl(coord.Envelope{T: coord.MsgEndpoints, Endpoints: eps})
	if err != nil {
		return err
	}
	return s.send(coord.FrameControl, payload)
}

func (s *Session) Close() {
	s.conn.Close()
}
