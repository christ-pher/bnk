// Package disco implements the path-discovery probe messages: pings that
// prove a candidate path works, pongs reporting the observed address, and
// call-me-maybe endpoint advertisements. Messages are sealed with a
// dedicated curve25519 keypair (never the WireGuard key) via nacl/box.
//
// Decryption alone does not authenticate the sender: callers must check
// the returned sender key against the known disco keys from the netmap.
package disco

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"net/netip"

	"golang.org/x/crypto/nacl/box"
)

// Magic prefixes every disco packet so the demux can classify traffic.
const Magic = "bnk\x00disco\x00"

// Wire layout: Magic (8) + sender pub (32) + nonce (24) + box(payload).
const headerLen = len(Magic) + 32 + 24

// Payload type tags.
const (
	typePing        = 1
	typePong        = 2
	typeCallMeMaybe = 3
)

type Message interface{ isMessage() }

type Ping struct {
	TxID [12]byte
}

type Pong struct {
	TxID     [12]byte
	Observed netip.AddrPort // where the ping appeared to come from
}

type CallMeMaybe struct {
	Endpoints []netip.AddrPort
}

func (Ping) isMessage()        {}
func (Pong) isMessage()        {}
func (CallMeMaybe) isMessage() {}

func NewKeypair() (priv, pub [32]byte, err error) {
	pubP, privP, err := box.GenerateKey(rand.Reader)
	if err != nil {
		return priv, pub, err
	}
	return *privP, *pubP, nil
}

// encodeAddrPort is IPv4-only, matching the v1 tunnel: 4 bytes IP + 2 port.
func encodeAddrPort(ap netip.AddrPort) []byte {
	a := ap.Addr().Unmap().As4()
	out := make([]byte, 6)
	copy(out, a[:])
	binary.BigEndian.PutUint16(out[4:], ap.Port())
	return out
}

func decodeAddrPort(b []byte) (netip.AddrPort, bool) {
	if len(b) < 6 {
		return netip.AddrPort{}, false
	}
	return netip.AddrPortFrom(netip.AddrFrom4([4]byte(b[:4])), binary.BigEndian.Uint16(b[4:6])), true
}

func encodePayload(msg Message) []byte {
	switch m := msg.(type) {
	case Ping:
		return append([]byte{typePing}, m.TxID[:]...)
	case Pong:
		out := append([]byte{typePong}, m.TxID[:]...)
		return append(out, encodeAddrPort(m.Observed)...)
	case CallMeMaybe:
		out := []byte{typeCallMeMaybe}
		for _, ep := range m.Endpoints {
			out = append(out, encodeAddrPort(ep)...)
		}
		return out
	}
	return nil
}

func decodePayload(b []byte) (Message, error) {
	if len(b) < 1 {
		return nil, fmt.Errorf("disco: empty payload")
	}
	switch b[0] {
	case typePing:
		if len(b) < 13 {
			return nil, fmt.Errorf("disco: short ping")
		}
		var p Ping
		copy(p.TxID[:], b[1:13])
		return p, nil
	case typePong:
		if len(b) < 19 {
			return nil, fmt.Errorf("disco: short pong")
		}
		var p Pong
		copy(p.TxID[:], b[1:13])
		ap, ok := decodeAddrPort(b[13:])
		if !ok {
			return nil, fmt.Errorf("disco: bad pong addr")
		}
		p.Observed = ap
		return p, nil
	case typeCallMeMaybe:
		var m CallMeMaybe
		for rest := b[1:]; len(rest) > 0; rest = rest[6:] {
			ap, ok := decodeAddrPort(rest)
			if !ok {
				return nil, fmt.Errorf("disco: bad call-me-maybe endpoint")
			}
			m.Endpoints = append(m.Endpoints, ap)
		}
		return m, nil
	}
	return nil, fmt.Errorf("disco: unknown payload type %d", b[0])
}

// Seal builds a full wire packet: Magic + sender pub + nonce + box(msg).
func Seal(msg Message, senderPriv, senderPub, recvPub [32]byte) []byte {
	var nonce [24]byte
	rand.Read(nonce[:])
	out := make([]byte, 0, headerLen+len(Magic))
	out = append(out, Magic...)
	out = append(out, senderPub[:]...)
	out = append(out, nonce[:]...)
	return box.Seal(out, encodePayload(msg), &nonce, &recvPub, &senderPriv)
}

// Open verifies and decrypts a disco packet, returning the sender's public
// disco key and the message.
func Open(pkt []byte, recvPriv [32]byte) ([32]byte, Message, error) {
	var sender [32]byte
	if !IsDisco(pkt) || len(pkt) < headerLen+box.Overhead {
		return sender, nil, fmt.Errorf("disco: not a complete disco packet")
	}
	copy(sender[:], pkt[len(Magic):len(Magic)+32])
	var nonce [24]byte
	copy(nonce[:], pkt[len(Magic)+32:headerLen])
	payload, ok := box.Open(nil, pkt[headerLen:], &nonce, &sender, &recvPriv)
	if !ok {
		return sender, nil, fmt.Errorf("disco: packet failed to authenticate")
	}
	msg, err := decodePayload(payload)
	if err != nil {
		return sender, nil, err
	}
	return sender, msg, nil
}

// IsDisco reports whether pkt begins with the disco magic.
func IsDisco(pkt []byte) bool {
	return len(pkt) >= len(Magic) && string(pkt[:len(Magic)]) == Magic
}
