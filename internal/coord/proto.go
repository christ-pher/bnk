// Package coord defines the client↔server coordination protocol: length-
// prefixed binary frames over one long-lived connection. Control frames
// carry JSON; relay frames carry raw encrypted WireGuard packets.
package coord

import (
	"encoding/json"
	"fmt"
	"io"
	"net/netip"

	"github.com/christ-pher/bnk/internal/netmap"
)

type FrameType byte

const (
	FrameControl FrameType = 1 + iota
	FrameRelayData
	FrameKeepalive
)

// MaxFrameSize bounds a frame payload: far above any netmap or relayed
// packet, far below the 2^24-1 the header could express.
const MaxFrameSize = 1 << 20

func WriteFrame(w io.Writer, typ FrameType, payload []byte) error {
	if len(payload) > MaxFrameSize {
		return fmt.Errorf("coord: frame payload %d bytes exceeds max %d", len(payload), MaxFrameSize)
	}
	hdr := [4]byte{byte(typ), byte(len(payload) >> 16), byte(len(payload) >> 8), byte(len(payload))}
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func ReadFrame(r io.Reader) (FrameType, []byte, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return 0, nil, err
	}
	size := int(hdr[1])<<16 | int(hdr[2])<<8 | int(hdr[3])
	if size > MaxFrameSize {
		return 0, nil, fmt.Errorf("coord: frame payload %d bytes exceeds max %d", size, MaxFrameSize)
	}
	payload := make([]byte, size)
	if _, err := io.ReadFull(r, payload); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return 0, nil, err
	}
	return FrameType(hdr[0]), payload, nil
}

// EnrollRequest/EnrollResponse are exchanged over plain HTTPS POST /enroll,
// outside the framed session, so enrollment stays curl-debuggable.
type EnrollRequest struct {
	EnrollKey string     `json:"enroll_key"`
	Hostname  string     `json:"hostname"`
	OS        string     `json:"os"`
	NodeKey   netmap.Key `json:"node_key"`
	DiscoKey  netmap.Key `json:"disco_key"`
}

type EnrollResponse struct {
	NodeID netmap.NodeID `json:"node_id"`
	IP     netip.Addr    `json:"ip"`
	Prefix netip.Prefix  `json:"prefix"`
}

// Message type tags for the control envelope.
const (
	MsgHello     = "hello"
	MsgChallenge = "challenge"
	MsgAuth      = "auth"
	MsgNetmap    = "netmap"
	MsgEndpoints = "endpoints"
	MsgDiscoFwd  = "disco_fwd"
	// MsgLeave asks the server to forget this node entirely. It is only
	// honored on an authenticated session, so possession of the node key
	// is already proven.
	MsgLeave = "leave"
	// MsgReject tells a client why the server will not open a session,
	// so it can report something actionable instead of retrying blindly.
	MsgReject = "reject"
)

// Envelope is the tagged union carried by control frames; T selects which
// pointer field is set.
type Envelope struct {
	T         string           `json:"t"`
	Hello     *Hello           `json:"hello,omitempty"`
	Challenge *Challenge       `json:"challenge,omitempty"`
	Auth      *Auth            `json:"auth,omitempty"`
	Netmap    *netmap.Netmap   `json:"netmap,omitempty"`
	Endpoints []netip.AddrPort `json:"endpoints,omitempty"`
	DiscoFwd  *DiscoFwd        `json:"disco_fwd,omitempty"`
	Reject    *Reject          `json:"reject,omitempty"`
}

// Reject explains a refused session.
type Reject struct {
	Reason string `json:"reason"`
}

type Hello struct {
	NodeKey netmap.Key `json:"node_key"`
}

// Challenge asks the connecting client to prove possession of the node
// private key: seal Value with nacl/box to EphPub using that key.
type Challenge struct {
	EphPub netmap.Key `json:"eph_pub"`
	Nonce  []byte     `json:"nonce"`
	Value  []byte     `json:"value"`
}

type Auth struct {
	Sealed []byte `json:"sealed"`
}

type DiscoFwd struct {
	Dst     netmap.NodeID `json:"dst"`
	Src     netmap.NodeID `json:"src,omitempty"`
	Payload []byte        `json:"payload"`
}

func EncodeControl(msg Envelope) ([]byte, error) {
	return json.Marshal(msg)
}

func DecodeControl(raw []byte) (Envelope, error) {
	var msg Envelope
	if err := json.Unmarshal(raw, &msg); err != nil {
		return Envelope{}, err
	}
	return msg, nil
}
