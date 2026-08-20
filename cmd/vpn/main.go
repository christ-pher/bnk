// vpn is the mesh client: it enrolls with a control server, brings up a
// WireGuard interface, and keeps the tunnel configured from pushed netmaps.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/signal"
	"sort"
	"syscall"
	"time"

	"golang.zx2c4.com/wireguard/tun"

	"vpnmesh/internal/cliutil"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/router"
	"vpnmesh/internal/vpnc"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "vpn:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vpn <up|down|status|ping|netcheck|run> [flags]")
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
	}
	if args[0] != "run" {
		return fmt.Errorf("usage: vpn run --server https://host:8443 [--key vpnkey:...] [flags]")
	}
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	serverURL := fs.String("server", "", "control server URL, e.g. https://vpn.example:8443")
	enrollKey := fs.String("key", "", "enrollment key (vpnkey:...), required on first run")
	stateDir := fs.String("state-dir", "/var/lib/vpn", "directory for client state")
	sock := socketFlag(fs)
	name := fs.String("name", defaultHostname(), "node name shown to the mesh")
	ifName := fs.String("ifname", "vpn0", "tunnel interface name")
	fs.Parse(args[1:])

	if err := os.MkdirAll(*stateDir, 0o700); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	err := vpnc.Run(ctx, vpnc.Config{
		ServerURL:  *serverURL,
		EnrollKey:  *enrollKey,
		StateDir:   *stateDir,
		SocketPath: *sock,
		Hostname:   *name,
		CreateTUN:  realTUN(*ifName),
		Logf:       log.Printf,
	})
	if ctx.Err() != nil {
		return nil // clean shutdown on signal
	}
	return err
}

// realTUN creates a kernel TUN device and applies OS addressing/routes.
func realTUN(ifName string) func(prefix netip.Prefix, mtu int) (tun.Device, func() error, error) {
	return func(prefix netip.Prefix, mtu int) (tun.Device, func() error, error) {
		dev, err := tun.CreateTUN(ifName, mtu)
		if err != nil {
			return nil, nil, fmt.Errorf("create tun (root required?): %w", err)
		}
		name, err := dev.Name()
		if err != nil {
			name = ifName
		}
		if err := router.Up(name, prefix, mtu); err != nil {
			dev.Close()
			return nil, nil, err
		}
		return dev, dev.Close, nil
	}
}

// defaultSocket is where the daemon serves the local API. It lives under
// /run (not the 0700 state dir) so status works without root.
const defaultSocket = "/run/vpnmesh/vpn.sock"

func socketFlag(fs *flag.FlagSet) *string {
	return fs.String("socket", defaultSocket, "path to the daemon's local API socket")
}

func localClient(sock string) *http.Client {
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", sock)
		},
	}}
}

func localGet(sock, path string, out any) error {
	resp, err := localClient(sock).Get("http://vpn" + path)
	if err != nil {
		return fmt.Errorf("is vpn up running? %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s", bytes.TrimSpace(msg))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// localPost sends a control verb to the daemon and decodes the reply.
func localPost(sock, path string) error {
	resp, err := localClient(sock).Post("http://vpn"+path, "application/json", nil)
	if err != nil {
		return fmt.Errorf("vpn daemon not running — start it with: systemctl start vpn (%w)", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s", bytes.TrimSpace(msg))
	}
	return nil
}

// upCmd tells the running daemon to (re)connect the tunnel.
func upCmd(args []string) error {
	fs := flag.NewFlagSet("up", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)
	if err := localPost(*sock, "/up"); err != nil {
		return err
	}
	var st vpnc.Status
	if err := localGet(*sock, "/status", &st); err == nil && st.Running {
		fmt.Printf("vpn is up: %s (%s)\n", st.Self.Name, st.Self.IP)
	} else {
		fmt.Println("vpn is up")
	}
	return nil
}

// downCmd tells the running daemon to tear the tunnel down (the daemon
// stays alive; `vpn up` reconnects).
func downCmd(args []string) error {
	fs := flag.NewFlagSet("down", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)
	if err := localPost(*sock, "/down"); err != nil {
		return err
	}
	fmt.Println("vpn is down (run `vpn up` to reconnect)")
	return nil
}

func pingCmd(args []string) error {
	fs := flag.NewFlagSet("ping", flag.ExitOnError)
	sock := socketFlag(fs)
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: vpn ping <peer-name>")
	}
	var out struct {
		Addr  string  `json:"addr"`
		RTTms float64 `json:"rtt_ms"`
	}
	if err := localGet(*sock, "/ping?peer="+url.QueryEscape(fs.Arg(0)), &out); err != nil {
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
	if err := localGet(*sock, "/netcheck", &out); err != nil {
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

	hc := localClient(*sock)
	resp, err := hc.Get("http://vpn/status")
	if err != nil {
		return fmt.Errorf("is vpn up running? %w", err)
	}
	defer resp.Body.Close()
	var st vpnc.Status
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		return err
	}
	if !st.Running {
		fmt.Println("vpn is down (run `vpn up` to connect)")
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
