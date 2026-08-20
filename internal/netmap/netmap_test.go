package netmap

import (
	"encoding/json"
	"net/netip"
	"strings"
	"testing"
)

func TestNetmapJSONRoundTrip(t *testing.T) {
	var nodeKey, discoKey Key
	nodeKey[0], discoKey[31] = 0xAB, 0xCD

	nm := Netmap{
		SelfID: 7,
		SelfIP: netip.MustParsePrefix("100.64.0.7/32"),
		Peers: []Peer{{
			ID:        3,
			Name:      "nas",
			NodeKey:   nodeKey,
			DiscoKey:  discoKey,
			IP:        netip.MustParseAddr("100.64.0.3"),
			Endpoints: []netip.AddrPort{netip.MustParseAddrPort("203.0.113.9:41641")},
			Online:    true,
			Tags:      []string{"servers"},
		}},
	}

	raw, err := json.Marshal(nm)
	if err != nil {
		t.Fatal(err)
	}
	var got Netmap
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.SelfID != nm.SelfID || got.SelfIP != nm.SelfIP {
		t.Errorf("self fields = %v %v, want %v %v", got.SelfID, got.SelfIP, nm.SelfID, nm.SelfIP)
	}
	if len(got.Peers) != 1 {
		t.Fatalf("peers = %+v", got.Peers)
	}
	p, q := got.Peers[0], nm.Peers[0]
	if p.NodeKey != q.NodeKey || p.DiscoKey != q.DiscoKey || p.IP != q.IP || p.Name != q.Name || !p.Online {
		t.Errorf("peer round-trip mismatch:\n got %+v\nwant %+v", p, q)
	}
	if len(p.Endpoints) != 1 || p.Endpoints[0] != q.Endpoints[0] {
		t.Errorf("endpoints = %v, want %v", p.Endpoints, q.Endpoints)
	}
}

func TestKeyMarshalsAsBase64String(t *testing.T) {
	var k Key
	k[0] = 0xFF
	raw, err := json.Marshal(k)
	if err != nil {
		t.Fatal(err)
	}
	s := string(raw)
	if !strings.HasPrefix(s, `"`) || strings.Contains(s, "[") {
		t.Errorf("Key marshals as %s, want a base64 JSON string", s)
	}
	if len(s) != 46 { // 44 base64 chars + 2 quotes
		t.Errorf("Key JSON length = %d (%s), want 46", len(s), s)
	}
}
