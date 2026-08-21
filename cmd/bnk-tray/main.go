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
	"errors"
	"flag"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"fyne.io/systray"

	"github.com/christ-pher/bnk/internal/localclient"
	"github.com/christ-pher/bnk/internal/selfupdate"
	"github.com/christ-pher/bnk/internal/trayui"
	"github.com/christ-pher/bnk/internal/vpnc"
)

//go:embed icons/connected.ico
var iconConnected []byte

//go:embed icons/disconnected.ico
var iconDisconnected []byte

const (
	pollInterval  = 3 * time.Second
	updateCheckIn = 6 * time.Hour
	repoURL       = "https://github.com/christ-pher/bnk"
)

// version is stamped by the release workflow; local builds say "dev".
var version = "dev"

var socket = flag.String("socket", vpnc.DefaultSocket, "daemon diagnostics pipe")

func main() {
	flag.Parse()
	startLogging()

	// One tray per session: clicking the Start menu entry again should
	// not leave a second icon behind, and two trays toggling the same
	// daemon only confuses whoever is looking at them.
	release, already := claimSingleInstance()
	if already {
		log.Printf("another bnk-tray is already running; exiting")
		return
	}
	defer release()

	log.Printf("bnk-tray %s starting (socket %s)", version, *socket)
	systray.Run(onReady, func() { log.Printf("bnk-tray exiting") })
}

// startLogging sends the tray's log to a file: a windowsgui binary has
// no console, so without this a failure leaves no trace at all.
func startLogging() {
	f, err := os.OpenFile(filepath.Join(os.TempDir(), "bnk-tray.log"),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return
	}
	log.SetOutput(f)
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
	update   *systray.MenuItem
	quit     *systray.MenuItem

	views chan trayui.View // rendered status from the poller
	wake  chan struct{}    // ask the poller for an immediate refresh

	mu         sync.Mutex
	lastView   trayui.View
	newVersion string
}

func onReady() {
	systray.SetIcon(iconDisconnected)
	systray.SetTitle("bnk")
	systray.SetTooltip("bnk")

	m := &menu{
		views: make(chan trayui.View, 4),
		wake:  make(chan struct{}, 4),
	}
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
	m.update = systray.AddMenuItem("Check for updates", "Ask GitHub for a newer release")
	m.quit = systray.AddMenuItem("Quit", "Close the tray (the VPN keeps running)")

	go m.poll()
	go m.run()
}

// poll owns every call to the daemon. It runs off the event loop so a
// slow or absent daemon can never stop a menu click being handled —
// which is what made Quit and Sign in appear dead.
func (m *menu) poll() {
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		st, err := localclient.Status(*socket)
		select {
		case m.views <- trayui.Build(st, err):
		default: // one is already queued; the newer one follows shortly
		}
		select {
		case <-ticker.C:
		case <-m.wake:
		}
	}
}

func (m *menu) refreshSoon() {
	select {
	case m.wake <- struct{}{}:
	default:
	}
}

// run is the event loop. Everything here must return promptly, so work
// that talks to the network is handed to a goroutine.
func (m *menu) run() {
	go m.checkForUpdate(false)
	updates := time.NewTicker(updateCheckIn)
	defer updates.Stop()

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
		case v := <-m.views:
			m.apply(v)
		case <-m.action.ClickedCh:
			go m.onAction()
		case <-m.copyIP.ClickedCh:
			if ip := m.view().SelfIP; ip != "" {
				setClipboard(ip)
			}
		case i := <-clicks:
			if peers := m.view().Peers; i < len(peers) {
				setClipboard(peers[i].IP)
			}
		case <-updates.C:
			go m.checkForUpdate(false)
		case <-m.update.ClickedCh:
			go m.onUpdateClicked()
		case <-m.quit.ClickedCh:
			log.Printf("quit clicked")
			systray.Quit()
			return
		}
	}
}

func (m *menu) view() trayui.View {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastView
}

// onAction runs the button whose meaning depends on the current state.
func (m *menu) onAction() {
	switch v := m.view(); {
	case v.Unreachable:
		m.startService()
	case v.NeedsJoin:
		m.signIn()
	default:
		m.toggle()
	}
}

func (m *menu) onUpdateClicked() {
	m.mu.Lock()
	newer := m.newVersion
	m.mu.Unlock()
	if newer != "" {
		m.status.SetTitle("Starting the installer…")
		if err := launchUpdater(newer); err != nil {
			alert("Could not start the installer:\n\n"+err.Error(), "bnk — update")
		}
		return
	}
	m.checkForUpdate(true)
}

// startService asks Windows to start the daemon, elevating on the way:
// starting a service needs privileges the tray deliberately does not
// hold, so this raises a consent prompt rather than failing quietly.
func (m *menu) startService() {
	exe, err := os.Executable()
	if err != nil {
		alert(err.Error(), "bnk")
		return
	}
	bnk := filepath.Join(filepath.Dir(exe), "bnk.exe")
	log.Printf("starting service via %s", bnk)
	if err := shellExecute("runas", bnk, "service start"); err != nil {
		alert("Could not start the bnk service:\n\n"+err.Error()+
			"\n\nIf it is not installed, reinstall bnk from the latest release.", "bnk")
		return
	}
	time.Sleep(2 * time.Second) // let the service come up before polling
	m.refreshSoon()
}

// signIn asks for a join command and hands it to the daemon.
func (m *menu) signIn() {
	log.Printf("sign-in: prompting")
	pasted, err := promptForJoin()
	if errors.Is(err, errCancelled) {
		log.Printf("sign-in: cancelled")
		return
	}
	if err != nil {
		log.Printf("sign-in: prompt failed: %v", err)
		alert(err.Error(), "bnk — sign in")
		return
	}
	join, perr := trayui.ParseJoin(pasted)
	if perr != nil {
		log.Printf("sign-in: parse failed: %v", perr)
		alert(perr.Error(), "bnk — sign in")
		return
	}

	m.status.SetTitle("Signing in…")
	log.Printf("sign-in: joining server=%q", join.Server)
	if err := localclient.Join(*socket, join.Server, join.Key); err != nil {
		log.Printf("sign-in: join failed: %v", err)
		alert("Sign-in failed:\n\n"+err.Error(), "bnk — sign in")
		m.refreshSoon()
		return
	}
	log.Printf("sign-in: joined")
	m.refreshSoon()
}

// toggle flips the tunnel and reports failures where they can be seen,
// which is where a non-operator account learns why nothing happened.
func (m *menu) toggle() {
	connected := m.view().Connected
	m.status.SetTitle("Working…")
	var err error
	if connected {
		err = localclient.Down(*socket)
	} else {
		err = localclient.Up(*socket)
	}
	if err != nil {
		log.Printf("toggle failed: %v", err)
		alert(err.Error(), "bnk")
	}
	m.refreshSoon()
}

// checkForUpdate asks GitHub what the latest release is. announce marks
// a check the user asked for, which should say something either way.
func (m *menu) checkForUpdate(announce bool) {
	if announce {
		m.update.SetTitle("Checking…")
	}
	latest, available, err := selfupdate.UpdateAvailable(repoURL, version)

	m.mu.Lock()
	if err == nil && available {
		m.newVersion = latest
	} else {
		m.newVersion = ""
	}
	m.mu.Unlock()

	switch {
	case err != nil:
		log.Printf("update check failed: %v", err)
		m.update.SetTitle("Check for updates")
		if announce {
			alert("Could not check for updates:\n\n"+err.Error(), "bnk — update")
		}
	case available:
		m.update.SetTitle("Update to " + latest)
	default:
		m.update.SetTitle("Up to date (" + version + ")")
	}
}

// apply renders one view. It runs only on the event loop, so menu items
// are never written from two goroutines at once.
func (m *menu) apply(v trayui.View) {
	m.mu.Lock()
	m.lastView = v
	m.mu.Unlock()

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
