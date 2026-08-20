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

	"vpnmesh/internal/coord"
	"vpnmesh/internal/netmap"
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

type Handlers struct {
	OnNetmap func(netmap.Netmap)
}

type Session struct {
	conn net.Conn
	wmu  sync.Mutex
	bw   *bufio.Writer
	done chan struct{}
}

// Done is closed when the session's read loop exits for any reason.
func (s *Session) Done() <-chan struct{} {
	return s.done
}

// Dial connects to the server's /c endpoint, upgrades to the framed
// protocol, sends hello, and starts the read loop. tlsConf nil means plain
// TCP (tests); production passes a fingerprint-pinning config.
func Dial(ctx context.Context, baseURL string, tlsConf *tls.Config, nodeKey netmap.Key, h Handlers) (*Session, error) {
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
	req := fmt.Sprintf("GET /c HTTP/1.1\r\nHost: %s\r\nConnection: Upgrade\r\nUpgrade: vpn-coord/1\r\n\r\n", u.Host)
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

	s := &Session{conn: conn, bw: bufio.NewWriter(conn), done: make(chan struct{})}
	hello, err := coord.EncodeControl(coord.Envelope{T: coord.MsgHello, Hello: &coord.Hello{NodeKey: nodeKey}})
	if err != nil {
		conn.Close()
		return nil, err
	}
	if err := s.send(coord.FrameControl, hello); err != nil {
		conn.Close()
		return nil, err
	}

	go s.readLoop(br, h)
	return s, nil
}

func (s *Session) readLoop(r *bufio.Reader, h Handlers) {
	defer close(s.done)
	defer s.conn.Close()
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
			if msg.T == coord.MsgNetmap && msg.Netmap != nil && h.OnNetmap != nil {
				h.OnNetmap(*msg.Netmap)
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
