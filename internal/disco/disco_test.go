package disco

import (
	"bytes"
	"net/netip"
	"testing"
)

func keys(t *testing.T) (aPriv, aPub, bPriv, bPub [32]byte) {
	t.Helper()
	var err error
	aPriv, aPub, err = NewKeypair()
	if err != nil {
		t.Fatal(err)
	}
	bPriv, bPub, err = NewKeypair()
	if err != nil {
		t.Fatal(err)
	}
	return
}

func TestSealOpenPingRoundTrip(t *testing.T) {
	aPriv, aPub, bPriv, bPub := keys(t)

	ping := Ping{TxID: [12]byte{1, 2, 3}}
	pkt := Seal(ping, aPriv, aPub, bPub)

	if !IsDisco(pkt) {
		t.Error("sealed packet not recognized as disco")
	}
	sender, msg, err := Open(pkt, bPriv)
	if err != nil {
		t.Fatal(err)
	}
	if sender != aPub {
		t.Error("sender pub mismatch")
	}
	got, ok := msg.(Ping)
	if !ok || got.TxID != ping.TxID {
		t.Errorf("msg = %#v, want %#v", msg, ping)
	}
}

func TestSealOpenPongAndCallMeMaybe(t *testing.T) {
	aPriv, aPub, bPriv, bPub := keys(t)
	_ = aPub

	pong := Pong{TxID: [12]byte{9}, Observed: netip.MustParseAddrPort("203.0.113.7:41641")}
	_, msg, err := Open(Seal(pong, aPriv, aPub, bPub), bPriv)
	if err != nil {
		t.Fatal(err)
	}
	if got := msg.(Pong); got.TxID != pong.TxID || got.Observed != pong.Observed {
		t.Errorf("pong = %#v, want %#v", got, pong)
	}

	cmm := CallMeMaybe{Endpoints: []netip.AddrPort{
		netip.MustParseAddrPort("192.168.1.5:41641"),
		netip.MustParseAddrPort("203.0.113.7:41641"),
	}}
	_, msg, err = Open(Seal(cmm, aPriv, aPub, bPub), bPriv)
	if err != nil {
		t.Fatal(err)
	}
	got := msg.(CallMeMaybe)
	if len(got.Endpoints) != 2 || got.Endpoints[0] != cmm.Endpoints[0] || got.Endpoints[1] != cmm.Endpoints[1] {
		t.Errorf("cmm = %#v, want %#v", got, cmm)
	}
}

func TestOpenRejectsTamperAndWrongRecipient(t *testing.T) {
	aPriv, aPub, bPriv, bPub := keys(t)
	_, _, cPriv, _ := keys(t)
	_ = bPriv

	pkt := Seal(Ping{TxID: [12]byte{5}}, aPriv, aPub, bPub)

	tampered := bytes.Clone(pkt)
	tampered[len(tampered)-1] ^= 0xFF
	if _, _, err := Open(tampered, bPriv); err == nil {
		t.Error("tampered packet opened")
	}

	if _, _, err := Open(pkt, cPriv); err == nil {
		t.Error("packet opened by non-recipient")
	}
}

func TestIsDiscoRejectsOtherTraffic(t *testing.T) {
	if IsDisco([]byte{0x45, 0x00, 0x00}) {
		t.Error("IPv4 packet classified as disco")
	}
	if IsDisco(nil) {
		t.Error("nil classified as disco")
	}
	// WireGuard packets start with type bytes 1-4 then zeros.
	if IsDisco([]byte{1, 0, 0, 0, 9, 9, 9, 9, 9, 9, 9, 9}) {
		t.Error("WireGuard-shaped packet classified as disco")
	}
}

func TestOpenRejectsTruncated(t *testing.T) {
	aPriv, aPub, _, bPub := keys(t)
	pkt := Seal(Ping{}, aPriv, aPub, bPub)
	for _, n := range []int{0, 7, len(Magic), len(Magic) + 31, len(pkt) - 17} {
		if _, _, err := Open(pkt[:n], [32]byte{}); err == nil {
			t.Errorf("truncated packet (%d bytes) opened", n)
		}
	}
}
