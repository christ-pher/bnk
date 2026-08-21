// bnk is the mesh client: it enrolls with a control server, brings up a
// WireGuard interface, and keeps the tunnel configured from pushed netmaps.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/netip"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"golang.zx2c4.com/wireguard/tun"

	"github.com/christ-pher/bnk/internal/cliutil"
	"github.com/christ-pher/bnk/internal/localclient"
	"github.com/christ-pher/bnk/internal/netmap"
	"github.com/christ-pher/bnk/internal/router"
	"github.com/christ-pher/bnk/internal/vpnc"
)

// version is stamped by the release workflow (-X main.version=vX.Y.Z);
// local builds report "dev".
var version = "dev"

const (
	repoURL         = "https://github.com/christ-pher/bnk"
	rawInstallerURL = "https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.ps1"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "bnk:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: bnk <up|down|status|join|leave|ping|netcheck|update|version|run|service> [flags]")
	}
	switch args[0] {
	case "status":
		return status(args[1:])
	case "ping":
		return pingCmd(args[1:])
	case "netcheck":
		return netcheckCmd(args[1:])
	case "up":
		return upCmd(args[1:])
	case "down":
		return downCmd(args[1:])
	case "version":
		fmt.Println(version)
		return nil
	case "update":
		return updateCmd()
	case "service":
		return serviceCmd(args[1:])
	case "leave":
		return leaveCmd(args[1:])
	case "join":
		return joinCmd(args[1:])
	}
	if args[0] != "run" {
		return fmt.Errorf("usage: bnk run --server https://host:8443 [--key bnkkey:...] [flags]")
	}
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	serverURL := fs.String("server", "", "control server URL, e.g. https://bnk.example:8443")
	enrollKey := fs.String("key", "", "enrollment key (bnkkey:...), required on first run")
	stateDir := fs.String("state-dir", vpnc.DefaultStateDir, "directory for client state")
	sock := socketFlag(fs)
	name := fs.String("name", defaultHostname(), "node name shown to the mesh")
	ifName := fs.String("ifname", "bnk0", "tunnel interface name")
	operator := fs.String("operator", "", "Windows only: SID of an account allowed to toggle the tunnel without elevating")
	fs.Parse(args[1:])

	if err := os.MkdirAll(*stateDir, 0o700); err != nil {
		return err
	}

	return runDaemon(vpnc.Config{
		ServerURL:   *serverURL,
		EnrollKey:   *enrollKey,
		StateDir:    *stateDir,
		SocketPath:  *sock,
		OperatorSID: *operator,
		Hostname:    *name,
		CreateTUN:   realTUN(*ifName),
		Logf:        log.Printf,
	})
}

// realTUN creates a kernel TUN device and applies OS addressing/routes.
func realTUN(ifName string) func(prefix netip.Prefix, mtu int) (tun.Device, func() error, error) {
	return func(prefix netip.Prefix, mtu int) (tun.Device, func() error, error) {
		dev, err := tun.CreateTUN(ifName, mtu)
		if err != nil {
			return nil, nil, fmt.Errorf("create tun (%w)", adminHint(err))
		}
		name, err := dev.Name()
		if err != nil {
			name = ifName
		}
		if err := router.Up(dev, name, prefix, mtu); err != nil {
			dev.Close()
			return nil, nil, err
		}
		return dev, dev.Close, nil
	}
}

// joinCmd enrols a running daemon into a mesh. It exists so a machine
// installed without a key — a plain double-click of the installer — can
// be signed in afterwards, from the command line or the tray.
func joinCmd(args []string) error {
	fs := flag.NewFlagSet("join", flag.ExitOnError)
	sock := socketFlag(fs)
	server := fs.String("server", "", "control server URL (optional if this node already knows it)")
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: bnk join [--server https://host:8443] <bnkkey:...>")
	}
	if err := localclient.Join(*sock, *server, fs.Arg(0)); err != nil {
		return err
	}
	var st vpnc.Status
	if err := localclient.Get(*sock, "/status", &st); err == nil && st.Running {
		fmt.Printf("joined: %s (%s)\n", st.Self.Name, st.Self.IP)
	} else {
		fmt.Println("joined")
	}
	return nil
}

// leaveCmd deregisters this node from the control server. The installers
// call it during uninstall so a removed machine does not linger in the
// server's node list.
func leaveCmd(args []string) error {
	fs := flag.NewFlagSet("leave", flag.ExitOnError)
	stateDir := fs.String("state-dir", vpnc.DefaultStateDir, "directory for client state")
	fs.Parse(args)
	if err := vpnc.Leave(context.Background(), *stateDir); err != nil {
		return fmt.Errorf("could not deregister (remove it on the server with `bnk-server node rm`): %w", err)
	}
	fmt.Println("left the mesh — the server no longer lists this node")
	return nil
}

func socketFlag(fs *flag.FlagSet) *string {
	return fs.String("socket", vpnc.DefaultSocket, "path to the daemon's local API socket")
}

// upCmd tells the running daemon to (re)connect the tunnel.
func upCmd(args []string) error {
	// Machines deployed before the daemon split had units running
	// `bnk up --server ... --key ...`; catch that with a pointer to the
	// fix instead of a flag-parse error.
	for _, a := range args {
		if strings.HasPrefix(a, "--server") || strings.HasPrefix(a, "--key") {
			return fmt.Errorf("the daemon now runs via `run`, not `up` — update the service (rerun the client install script), then use plain `up`/`down`")
		}
	}
	fs := flag.NewFlagSet("up", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)
	if err := localclient.Up(*sock); err != nil {
		return err
	}
	var st vpnc.Status
	if err := localclient.Get(*sock, "/status", &st); err == nil && st.Running {
		fmt.Printf("bnk is up: %s (%s)\n", st.Self.Name, st.Self.IP)
	} else {
		fmt.Println("bnk is up")
	}
	return nil
}

// downCmd tells the running daemon to tear the tunnel down (the daemon
// stays alive; `bnk up` reconnects).
func downCmd(args []string) error {
	fs := flag.NewFlagSet("down", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)
	if err := localclient.Down(*sock); err != nil {
		return err
	}
	fmt.Println("bnk is down (run `bnk up` to reconnect)")
	return nil
}

func pingCmd(args []string) error {
	fs := flag.NewFlagSet("ping", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: bnk ping <peer-name>")
	}
	var out struct {
		Addr  string  `json:"addr"`
		RTTms float64 `json:"rtt_ms"`
	}
	if err := localclient.Get(*sock, "/ping?peer="+url.QueryEscape(fs.Arg(0)), &out); err != nil {
		return err
	}
	fmt.Printf("pong from %s via %s: %.2fms (direct path proven)\n", fs.Arg(0), out.Addr, out.RTTms)
	return nil
}

func netcheckCmd(args []string) error {
	fs := flag.NewFlagSet("netcheck", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)
	var out map[string]any
	if err := localclient.Get(*sock, "/netcheck", &out); err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func status(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)

	var st vpnc.Status
	if err := localclient.Get(*sock, "/status", &st); err != nil {
		return err
	}
	if !st.Running {
		fmt.Println("bnk is down (run `bnk up` to connect)")
		return nil
	}
	cliutil.Table(os.Stdout, []string{"NODE", "IP", "ONLINE", "PATH", "LAST HANDSHAKE"}, statusRows(st, time.Now()))
	return nil
}

// statusRows merges self into the peer list (marked with *), ordered by
// node ID so the table is stable across runs.
func statusRows(st vpnc.Status, now time.Time) [][]string {
	type idRow struct {
		id  netmap.NodeID
		row []string
	}
	all := []idRow{{st.Self.ID, []string{st.Self.Name + "*", st.Self.IP.String(), "true", "-", "-"}}}
	for _, p := range st.Peers {
		hs := "never"
		if !p.LastHandshake.IsZero() {
			hs = now.Sub(p.LastHandshake).Round(time.Second).String() + " ago"
		}
		all = append(all, idRow{p.ID, []string{p.Name, p.IP.String(), fmt.Sprintf("%v", p.Online), p.Path, hs}})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].id < all[j].id })
	rows := make([][]string, len(all))
	for i, r := range all {
		rows[i] = r.row
	}
	return rows
}

func defaultHostname() string {
	h, err := os.Hostname()
	if err != nil {
		return "node"
	}
	return h
}
