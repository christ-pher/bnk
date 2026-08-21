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

// View is everything the tray renders for one status poll.
type View struct {
	// Title is the status line in the menu. It is kept short: the menu
	// is only as wide as its widest item, and this line is the widest.
	// Detail that does not earn that width lives in Tooltip instead.
	Title     string
	Tooltip   string // hover text, where the long form belongs
	Action    string // label of the connect-or-disconnect item
	Connected bool
	// NeedsJoin means the action item should ask for a key rather than
	// toggle a tunnel this machine cannot yet have.
	NeedsJoin bool
	SelfIP    string // empty when down; what "Copy my IP" yields
	Peers     []PeerRow
	Overflow  string // "…and N more", empty when everything fits
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
		return View{
			Title:   "Daemon not running",
			Tooltip: "bnk — the daemon is not reachable",
			Action:  "Connect",
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
		}
	}
	if !st.Running {
		return View{
			Title:   "Disconnected",
			Tooltip: "bnk — disconnected",
			Action:  "Connect",
		}
	}

	v := View{
		Title:     fmt.Sprintf("Connected — %s", st.Self.IP),
		Tooltip:   fmt.Sprintf("bnk — connected as %s (%s)", st.Self.Name, st.Self.IP),
		Action:    "Disconnect",
		Connected: true,
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
