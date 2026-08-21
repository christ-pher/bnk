// Package trayui turns daemon status into the strings the Windows tray
// menu shows. It is separated from the tray binary so the formatting is
// testable on any platform — the GUI itself is not.
package trayui

import (
	"fmt"
	"sort"

	"github.com/christ-pher/bnk/internal/vpnc"
)

// MaxPeerRows bounds the Peers submenu: systray cannot delete menu
// items, so a fixed pool is allocated once and rewritten on each poll.
const MaxPeerRows = 16

// Icon names the tray image a view calls for. The tray binary maps
// these onto the artwork in internal/trayicon; the choice lives here so
// it is decided in one place and tested on any platform.
type Icon int

const (
	// IconDisconnected is the tunnel being down on purpose.
	IconDisconnected Icon = iota
	// IconConnected is the tunnel being up.
	IconConnected
	// IconAttention is the tray needing something from the user before
	// either of the above is even possible.
	IconAttention
)

func (i Icon) String() string {
	switch i {
	case IconConnected:
		return "connected"
	case IconAttention:
		return "attention"
	}
	return "disconnected"
}

// View is everything the tray renders for one status poll.
type View struct {
	// Title is the status line in the menu. It is kept short: the menu
	// is only as wide as its widest item, and this line is the widest.
	// Detail that does not earn that width lives in Tooltip instead.
	Title     string
	Tooltip   string // hover text, where the long form belongs
	Action    string // label of the connect-or-disconnect item
	Connected bool
	// Icon is which tray image to show. It is not derived from
	// Connected alone: "off" and "cannot work yet" look the same in a
	// two-icon scheme, and only one of them wants the user to act.
	Icon Icon
	// NeedsJoin means the action item should ask for a key rather than
	// toggle a tunnel this machine cannot yet have.
	NeedsJoin bool
	// Unreachable means the daemon did not answer at all, so neither
	// toggling nor signing in can work yet.
	Unreachable bool
	SelfIP      string // empty when down; what "Copy my IP" yields
	Peers       []PeerRow
	Overflow    string // "…and N more", empty when everything fits
}

// PeerRow is one entry in the Peers submenu.
type PeerRow struct {
	Label string
	IP    string // what clicking the row copies
}

// Build renders a view. err is whatever the last poll returned, so the
// tray can explain itself when the daemon is unreachable.
func Build(st vpnc.Status, err error) View {
	if err != nil {
		// Offering "Connect" here was misleading: there is nothing to
		// connect to. Offer the thing that would actually help — the
		// tray elevates to do it, since starting a service needs
		// privileges it deliberately does not hold.
		return View{
			Title:       "Not running",
			Tooltip:     "bnk — the background service is not running",
			Action:      "Start the bnk service",
			Unreachable: true,
			Icon:        IconAttention,
		}
	}
	if !st.Enrolled {
		// Installed but never given a key: connecting is meaningless
		// until the machine knows which mesh it belongs to.
		return View{
			Title:     "Not signed in",
			Tooltip:   "bnk — not signed in to a mesh",
			Action:    "Sign in…",
			NeedsJoin: true,
			Icon:      IconAttention,
		}
	}
	if !st.Running {
		return View{
			Title:   "Disconnected",
			Tooltip: "bnk — disconnected",
			Action:  "Connect",
			Icon:    IconDisconnected,
		}
	}

	v := View{
		Title:     fmt.Sprintf("Connected — %s", st.Self.IP),
		Tooltip:   fmt.Sprintf("bnk — connected as %s (%s)", st.Self.Name, st.Self.IP),
		Action:    "Disconnect",
		Connected: true,
		Icon:      IconConnected,
		SelfIP:    st.Self.IP.String(),
	}

	peers := append([]vpnc.PeerStatus(nil), st.Peers...)
	// Online first, then by name: the reachable ones are what you act on.
	sort.Slice(peers, func(i, j int) bool {
		if peers[i].Online != peers[j].Online {
			return peers[i].Online
		}
		return peers[i].Name < peers[j].Name
	})

	for i, p := range peers {
		if i >= MaxPeerRows {
			v.Overflow = fmt.Sprintf("…and %d more", len(peers)-MaxPeerRows)
			break
		}
		v.Peers = append(v.Peers, PeerRow{Label: peerLabel(p), IP: p.IP.String()})
	}
	return v
}

// peerLabel reads as a status at a glance: a marker, the name, its
// address, and how traffic reaches it.
func peerLabel(p vpnc.PeerStatus) string {
	marker, state := "○", "offline"
	if p.Online {
		marker = "●"
		state = p.Path // "direct" or "relay"
	}
	return fmt.Sprintf("%s %s  %s  %s", marker, p.Name, p.IP, state)
}
