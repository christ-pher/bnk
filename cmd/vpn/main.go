// vpn is the mesh client: it enrolls with a control server, brings up a
// WireGuard interface, and keeps the tunnel configured from pushed netmaps.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/netip"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"text/tabwriter"
	"time"

	"golang.zx2c4.com/wireguard/tun"

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
		return fmt.Errorf("usage: vpn <up|status> [flags]")
	}
	if args[0] == "status" {
		return status(args[1:])
	}
	if args[0] != "up" {
		return fmt.Errorf("usage: vpn up --server https://host:8443 [--key vpnkey:...] [flags]")
	}
	fs := flag.NewFlagSet("up", flag.ExitOnError)
	serverURL := fs.String("server", "", "control server URL, e.g. https://vpn.example:8443")
	enrollKey := fs.String("key", "", "enrollment key (vpnkey:...), required on first run")
	stateDir := fs.String("state-dir", "/var/lib/vpn", "directory for client state")
	name := fs.String("name", defaultHostname(), "node name shown to the mesh")
	ifName := fs.String("ifname", "vpn0", "tunnel interface name")
	fs.Parse(args[1:])

	if err := os.MkdirAll(*stateDir, 0o700); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	err := vpnc.Run(ctx, vpnc.Config{
		ServerURL: *serverURL,
		EnrollKey: *enrollKey,
		StateDir:  *stateDir,
		Hostname:  *name,
		CreateTUN: realTUN(*ifName),
		Logf:      log.Printf,
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

func status(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	stateDir := fs.String("state-dir", "/var/lib/vpn", "directory for client state")
	fs.Parse(args)

	sock := filepath.Join(*stateDir, "vpn.sock")
	hc := &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", sock)
		},
	}}
	resp, err := hc.Get("http://vpn/status")
	if err != nil {
		return fmt.Errorf("is vpn up running? %w", err)
	}
	defer resp.Body.Close()
	var st vpnc.Status
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		return err
	}

	fmt.Printf("self: node %d, %s\n", st.Self.ID, st.Self.IP)
	tw := tabwriter.NewWriter(os.Stdout, 2, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "PEER\tIP\tONLINE\tPATH\tLAST HANDSHAKE")
	for _, p := range st.Peers {
		hs := "never"
		if !p.LastHandshake.IsZero() {
			hs = time.Since(p.LastHandshake).Round(time.Second).String() + " ago"
		}
		fmt.Fprintf(tw, "%s\t%s\t%v\t%s\t%s\n", p.Name, p.IP, p.Online, p.Path, hs)
	}
	return tw.Flush()
}

func defaultHostname() string {
	h, err := os.Hostname()
	if err != nil {
		return "node"
	}
	return h
}
