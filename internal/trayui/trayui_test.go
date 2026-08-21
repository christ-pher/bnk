package trayui_test

import (
	"errors"
	"net/netip"
	"strings"
	"testing"

	"github.com/christ-pher/bnk/internal/trayui"
	"github.com/christ-pher/bnk/internal/vpnc"
)

func peer(name, ip string, online bool, path string) vpnc.PeerStatus {
	return vpnc.PeerStatus{Name: name, IP: netip.MustParseAddr(ip), Online: online, Path: path}
}

func connected() vpnc.Status {
	return vpnc.Status{
		Running:  true,
		Enrolled: true,
		Self:     vpnc.SelfStatus{Name: "desktop", IP: netip.MustParseAddr("100.64.0.4")},
		Peers: []vpnc.PeerStatus{
			peer("zulu", "100.64.0.9", false, "relay"),
			peer("alpha", "100.64.0.1", true, "direct"),
			peer("beta", "100.64.0.2", true, "relay"),
		},
	}
}

func TestBuildWhenConnected(t *testing.T) {
	v := trayui.Build(connected(), nil)
	if !v.Connected {
		t.Error("view should be connected")
	}
	if v.Action != "Disconnect" {
		t.Errorf("action = %q, want Disconnect", v.Action)
	}
	if !strings.Contains(v.Title, "100.64.0.4") {
		t.Errorf("title %q should name the address", v.Title)
	}
	// The menu is only as wide as its widest item, so the status line
	// stays short and the hostname lives in the tooltip.
	if len(v.Title) > 30 {
		t.Errorf("title %q is %d chars; it drives the menu width", v.Title, len(v.Title))
	}
	if !strings.Contains(v.Tooltip, "desktop") {
		t.Errorf("tooltip %q should carry the node name", v.Tooltip)
	}
	if v.SelfIP != "100.64.0.4" {
		t.Errorf("SelfIP = %q", v.SelfIP)
	}
}

// Online peers come first so the reachable ones are at the top.
func TestBuildOrdersOnlinePeersFirstThenByName(t *testing.T) {
	v := trayui.Build(connected(), nil)
	if len(v.Peers) != 3 {
		t.Fatalf("got %d peer rows, want 3", len(v.Peers))
	}
	wantOrder := []string{"alpha", "beta", "zulu"}
	for i, want := range wantOrder {
		if !strings.Contains(v.Peers[i].Label, want) {
			t.Errorf("row %d = %q, want it to be %s", i, v.Peers[i].Label, want)
		}
	}
	if !strings.Contains(v.Peers[0].Label, "direct") {
		t.Errorf("online peer row %q should show its path", v.Peers[0].Label)
	}
	if !strings.Contains(v.Peers[2].Label, "offline") {
		t.Errorf("offline peer row %q should say offline", v.Peers[2].Label)
	}
	if v.Peers[1].IP != "100.64.0.2" {
		t.Errorf("row IP = %q, want the peer address for copying", v.Peers[1].IP)
	}
}

func TestBuildWhenDown(t *testing.T) {
	v := trayui.Build(vpnc.Status{Running: false, Enrolled: true}, nil)
	if v.Connected {
		t.Error("view should not be connected")
	}
	if v.Action != "Connect" {
		t.Errorf("action = %q, want Connect", v.Action)
	}
	if len(v.Peers) != 0 {
		t.Error("a disconnected view should list no peers")
	}
}

// An unreachable daemon must say so rather than looking disconnected.
func TestBuildWhenDaemonUnreachable(t *testing.T) {
	v := trayui.Build(vpnc.Status{}, errors.New("dial: no such file"))
	if v.Action != "Start the bnk service" {
		t.Errorf("action = %q, want an offer to start the service", v.Action)
	}
	if !strings.Contains(v.Title, "Not running") {
		t.Errorf("title = %q, want it to say the daemon is not running", v.Title)
	}
}

// systray cannot delete menu items, so the row pool is fixed and the
// remainder has to be summarised.
func TestBuildSummarisesPeersBeyondThePool(t *testing.T) {
	st := vpnc.Status{Running: true, Enrolled: true, Self: vpnc.SelfStatus{Name: "self", IP: netip.MustParseAddr("100.64.0.1")}}
	for i := 0; i < trayui.MaxPeerRows+3; i++ {
		st.Peers = append(st.Peers, peer("node", "100.64.1.1", true, "direct"))
	}
	v := trayui.Build(st, nil)
	if len(v.Peers) != trayui.MaxPeerRows {
		t.Errorf("got %d rows, want the pool size %d", len(v.Peers), trayui.MaxPeerRows)
	}
	if v.Overflow != "…and 3 more" {
		t.Errorf("overflow = %q, want \"…and 3 more\"", v.Overflow)
	}
}

// A machine that was installed but never given a key needs a different
// offer: connecting is meaningless until it knows which mesh to join.
func TestBuildWhenNotEnrolled(t *testing.T) {
	v := trayui.Build(vpnc.Status{Enrolled: false}, nil)
	if !v.NeedsJoin {
		t.Error("view should ask for sign-in")
	}
	if v.Action != "Sign in…" {
		t.Errorf("action = %q, want a sign-in prompt", v.Action)
	}
	if v.Connected {
		t.Error("an unenrolled node is not connected")
	}
	if !strings.Contains(v.Title, "Not signed in") {
		t.Errorf("title = %q", v.Title)
	}
}
