//go:build windows

// bnk-tray is the Windows system tray control for the bnk client. It
// shows the mesh at a glance and toggles the tunnel without an elevated
// prompt — the daemon grants that to the configured operator account.
//
// It is a separate binary from bnk.exe because the -H=windowsgui link
// flag that suppresses the console window applies to the whole binary
// and would silence the CLI.
package main

import (
	_ "embed"
	"flag"
	"time"

	"fyne.io/systray"

	"github.com/christ-pher/bnk/internal/localclient"
	"github.com/christ-pher/bnk/internal/trayui"
	"github.com/christ-pher/bnk/internal/vpnc"
)

//go:embed icons/connected.ico
var iconConnected []byte

//go:embed icons/disconnected.ico
var iconDisconnected []byte

const pollInterval = 3 * time.Second

var socket = flag.String("socket", vpnc.DefaultSocket, "daemon diagnostics pipe")

func main() {
	flag.Parse()
	systray.Run(onReady, func() {})
}

// menu holds the fixed set of items; systray cannot delete items, so
// everything is created once and rewritten as status changes.
type menu struct {
	status   *systray.MenuItem
	action   *systray.MenuItem
	peers    *systray.MenuItem
	peerRows []*systray.MenuItem
	overflow *systray.MenuItem
	copyIP   *systray.MenuItem
	quit     *systray.MenuItem

	// lastView backs the click handlers, which need the addresses that
	// were on screen when the user clicked.
	lastView trayui.View
}

func onReady() {
	systray.SetIcon(iconDisconnected)
	systray.SetTitle("bnk")
	systray.SetTooltip("bnk")

	m := &menu{}
	m.status = systray.AddMenuItem("Starting…", "")
	m.status.Disable()
	systray.AddSeparator()
	m.action = systray.AddMenuItem("Connect", "Connect or disconnect the tunnel")
	m.peers = systray.AddMenuItem("Peers", "Nodes in the mesh")
	for i := 0; i < trayui.MaxPeerRows; i++ {
		item := m.peers.AddSubMenuItem("", "Click to copy this address")
		item.Hide()
		m.peerRows = append(m.peerRows, item)
	}
	m.overflow = m.peers.AddSubMenuItem("", "")
	m.overflow.Disable()
	m.overflow.Hide()
	m.copyIP = systray.AddMenuItem("Copy my IP", "Copy this machine's mesh address")
	systray.AddSeparator()
	m.quit = systray.AddMenuItem("Quit", "Close the tray (the VPN keeps running)")

	go m.run()
}

func (m *menu) run() {
	m.refresh()
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	clicks := make(chan int, 8)
	for i, row := range m.peerRows {
		go func(i int, ch <-chan struct{}) {
			for range ch {
				clicks <- i
			}
		}(i, row.ClickedCh)
	}

	for {
		select {
		case <-ticker.C:
			m.refresh()
		case <-m.action.ClickedCh:
			m.toggle()
		case <-m.copyIP.ClickedCh:
			if ip := m.lastView.SelfIP; ip != "" {
				setClipboard(ip)
			}
		case i := <-clicks:
			if i < len(m.lastView.Peers) {
				setClipboard(m.lastView.Peers[i].IP)
			}
		case <-m.quit.ClickedCh:
			systray.Quit()
			return
		}
	}
}

// toggle flips the tunnel and reports failures in the status line, which
// is where a non-operator account learns why nothing happened.
func (m *menu) toggle() {
	m.status.SetTitle("Working…")
	var err error
	if m.lastView.Connected {
		err = localclient.Down(*socket)
	} else {
		err = localclient.Up(*socket)
	}
	if err != nil {
		m.status.SetTitle(truncate("Failed: " + err.Error()))
		return
	}
	m.refresh()
}

func (m *menu) refresh() {
	st, err := localclient.Status(*socket)
	v := trayui.Build(st, err)
	m.lastView = v

	m.status.SetTitle(v.Title)
	systray.SetTooltip(v.Tooltip)
	m.action.SetTitle(v.Action)
	if v.Connected {
		systray.SetIcon(iconConnected)
	} else {
		systray.SetIcon(iconDisconnected)
	}
	if v.SelfIP == "" {
		m.copyIP.Hide()
	} else {
		m.copyIP.Show()
	}

	for i, row := range m.peerRows {
		if i < len(v.Peers) {
			row.SetTitle(v.Peers[i].Label)
			row.Show()
		} else {
			row.Hide()
		}
	}
	if v.Overflow == "" {
		m.overflow.Hide()
	} else {
		m.overflow.SetTitle(v.Overflow)
		m.overflow.Show()
	}
	if len(v.Peers) == 0 {
		m.peers.Disable()
	} else {
		m.peers.Enable()
	}
}

// truncate keeps a long error from stretching the menu off screen.
func truncate(s string) string {
	const max = 70
	if len(s) <= max {
		return s
	}
	return s[:max-1] + "…"
}
