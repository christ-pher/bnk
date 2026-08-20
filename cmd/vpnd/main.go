// vpnd is the control server: it enrolls nodes, assigns IPs, pushes
// netmaps, and (soon) relays traffic. Admin subcommands talk to a running
// vpnd over a unix socket in the state directory.
package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"vpnmesh/internal/cliutil"
	"vpnmesh/internal/coord/server"
	"vpnmesh/internal/pin"
	"vpnmesh/internal/store"
	"vpnmesh/internal/stunner"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "vpnd:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vpnd <up|down|serve|key|node|acl> [flags]")
	}
	switch args[0] {
	case "serve":
		return serve(args[1:])
	case "up":
		return systemctl("start")
	case "down":
		return systemctl("stop")
	case "key":
		if len(args) < 2 {
			return fmt.Errorf("usage: vpnd key <new|ls|revoke> ...")
		}
		switch args[1] {
		case "new":
			return keyNew(args[2:])
		case "ls":
			return keyLs(args[2:])
		case "revoke":
			return keyRevoke(args[2:])
		default:
			return fmt.Errorf("usage: vpnd key <new|ls|revoke> ...")
		}
	case "node":
		if len(args) < 2 || args[1] != "ls" {
			return fmt.Errorf("usage: vpnd node ls")
		}
		return nodeLs(args[2:])
	case "acl":
		if len(args) < 2 {
			return fmt.Errorf("usage: vpnd acl <set|get|check> ...")
		}
		switch args[1] {
		case "set":
			return aclSet(args[2:])
		case "get":
			return aclGet(args[2:])
		case "check":
			return aclCheck(args[2:])
		default:
			return fmt.Errorf("usage: vpnd acl <set|get|check> ...")
		}
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func stateDirFlag(fs *flag.FlagSet) *string {
	return fs.String("state-dir", "/var/lib/vpnd", "directory for state, cert, and admin socket")
}

// systemctl starts or stops the vpnd service: `vpnd up` / `vpnd down`
// control the control server without remembering systemd incantations.
func systemctl(verb string) error {
	cmd := exec.Command("systemctl", verb, "vpnd")
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("systemctl %s vpnd: %w (is the vpnd service installed? see install-server.sh)", verb, err)
	}
	if verb == "start" {
		fmt.Println("vpnd is up")
	} else {
		fmt.Println("vpnd is down (run `vpnd up` to start it again)")
	}
	return nil
}

func serve(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	listen := fs.String("listen", ":8443", "TLS listen address for clients")
	fs.Parse(args)

	if err := os.MkdirAll(*stateDir, 0o700); err != nil {
		return err
	}
	certPEM, keyPEM, fp, err := loadOrCreateCert(*stateDir)
	if err != nil {
		return err
	}
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return err
	}

	srv, err := server.New(store.NewFileStore(filepath.Join(*stateDir, "state.json")))
	if err != nil {
		return err
	}

	// STUN responder on the same port number, UDP: one port to open.
	stunPC, err := net.ListenPacket("udp", *listen)
	if err != nil {
		return err
	}
	go stunner.Serve(context.Background(), stunPC)

	sockPath := filepath.Join(*stateDir, "admin.sock")
	os.Remove(sockPath)
	adminLn, err := net.Listen("unix", sockPath)
	if err != nil {
		return err
	}
	go http.Serve(adminLn, srv.AdminHandler(fp))

	fmt.Printf("vpnd: listening on %s\n", *listen)
	fmt.Printf("vpnd: cert fingerprint %s\n", fp)
	fmt.Printf("vpnd: mint enrollment keys with: vpnd key new --state-dir %s\n", *stateDir)

	hs := &http.Server{
		Addr:      *listen,
		Handler:   srv.Handler(),
		TLSConfig: &tls.Config{Certificates: []tls.Certificate{cert}},
	}
	return hs.ListenAndServeTLS("", "")
}

func loadOrCreateCert(dir string) (certPEM, keyPEM []byte, fp string, err error) {
	certPath, keyPath := filepath.Join(dir, "cert.pem"), filepath.Join(dir, "key.pem")
	certPEM, cerr := os.ReadFile(certPath)
	keyPEM, kerr := os.ReadFile(keyPath)
	if cerr != nil || kerr != nil {
		certPEM, keyPEM, err = pin.GenerateCert()
		if err != nil {
			return nil, nil, "", err
		}
		if err := os.WriteFile(certPath, certPEM, 0o644); err != nil {
			return nil, nil, "", err
		}
		if err := os.WriteFile(keyPath, keyPEM, 0o600); err != nil {
			return nil, nil, "", err
		}
	}
	fp, err = pin.Fingerprint(certPEM)
	return certPEM, keyPEM, fp, err
}

// adminClient speaks HTTP over the unix admin socket.
func adminClient(stateDir string) *http.Client {
	sock := filepath.Join(stateDir, "admin.sock")
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", sock)
		},
	}}
}

func keyNew(args []string) error {
	fs := flag.NewFlagSet("key new", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	reusable := fs.Bool("reusable", false, "allow the key to enroll multiple nodes")
	ttl := fs.Duration("ttl", 24*time.Hour, "how long the key stays valid")
	fs.Parse(args)

	u := fmt.Sprintf("http://vpnd/enroll-keys?ttl=%s&reusable=%v", url.QueryEscape(ttl.String()), *reusable)
	resp, err := adminClient(*stateDir).Post(u, "application/json", nil)
	if err != nil {
		return fmt.Errorf("is vpnd running? %w", err)
	}
	defer resp.Body.Close()
	var out struct {
		Key string `json:"key"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	fmt.Println(out.Key)
	return nil
}

func aclSet(args []string) error {
	fs := flag.NewFlagSet("acl set", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: vpnd acl set <policy.json>")
	}
	policy, err := os.ReadFile(fs.Arg(0))
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPut, "http://vpnd/policy", bytes.NewReader(policy))
	if err != nil {
		return err
	}
	resp, err := adminClient(*stateDir).Do(req)
	if err != nil {
		return fmt.Errorf("is vpnd running? %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("policy rejected: %s", bytes.TrimSpace(msg))
	}
	fmt.Println("policy applied")
	return nil
}

func aclGet(args []string) error {
	fs := flag.NewFlagSet("acl get", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)
	resp, err := adminClient(*stateDir).Get("http://vpnd/policy")
	if err != nil {
		return fmt.Errorf("is vpnd running? %w", err)
	}
	defer resp.Body.Close()
	_, err = io.Copy(os.Stdout, resp.Body)
	return err
}

func aclCheck(args []string) error {
	fs := flag.NewFlagSet("acl check", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)
	if fs.NArg() != 3 {
		return fmt.Errorf("usage: vpnd acl check <src-node> <dst-node> <tcp/22|udp/53|icmp>")
	}
	u := fmt.Sprintf("http://vpnd/check?src=%s&dst=%s&target=%s",
		url.QueryEscape(fs.Arg(0)), url.QueryEscape(fs.Arg(1)), url.QueryEscape(fs.Arg(2)))
	resp, err := adminClient(*stateDir).Get(u)
	if err != nil {
		return fmt.Errorf("is vpnd running? %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s", bytes.TrimSpace(msg))
	}
	var verdict struct {
		Allowed bool `json:"allowed"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&verdict); err != nil {
		return err
	}
	if verdict.Allowed {
		fmt.Println("allowed")
	} else {
		fmt.Println("denied")
	}
	return nil
}

func keyLs(args []string) error {
	fs := flag.NewFlagSet("key ls", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)

	resp, err := adminClient(*stateDir).Get("http://vpnd/enroll-keys")
	if err != nil {
		return fmt.Errorf("is vpnd running? %w", err)
	}
	defer resp.Body.Close()
	var keys []server.AdminKey
	if err := json.NewDecoder(resp.Body).Decode(&keys); err != nil {
		return err
	}
	rows := make([][]string, 0, len(keys))
	for _, k := range keys {
		exp := "never"
		if !k.ExpiresAt.IsZero() {
			exp = k.ExpiresAt.Local().Format("2006-01-02 15:04")
			if time.Now().After(k.ExpiresAt) {
				exp += " (expired)"
			}
		}
		rows = append(rows, []string{k.Prefix, fmt.Sprintf("%v", k.Reusable), fmt.Sprintf("%v", k.Used), fmt.Sprintf("%v", k.Revoked), exp})
	}
	cliutil.Table(os.Stdout, []string{"PREFIX", "REUSABLE", "USED", "REVOKED", "EXPIRES"}, rows)
	return nil
}

func keyRevoke(args []string) error {
	fs := flag.NewFlagSet("key revoke", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: vpnd key revoke <prefix>")
	}
	resp, err := adminClient(*stateDir).Post("http://vpnd/enroll-keys/revoke?prefix="+url.QueryEscape(fs.Arg(0)), "application/json", nil)
	if err != nil {
		return fmt.Errorf("is vpnd running? %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s", bytes.TrimSpace(msg))
	}
	fmt.Println("revoked")
	return nil
}

func nodeLs(args []string) error {
	fs := flag.NewFlagSet("node ls", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)

	resp, err := adminClient(*stateDir).Get("http://vpnd/nodes")
	if err != nil {
		return fmt.Errorf("is vpnd running? %w", err)
	}
	defer resp.Body.Close()
	var nodes []server.AdminNode
	if err := json.NewDecoder(resp.Body).Decode(&nodes); err != nil {
		return err
	}
	rows := make([][]string, 0, len(nodes))
	for _, n := range nodes {
		rows = append(rows, []string{fmt.Sprint(n.ID), n.Name, n.IP.String(), n.OS, fmt.Sprintf("%v", n.Online)})
	}
	cliutil.Table(os.Stdout, []string{"ID", "NAME", "IP", "OS", "ONLINE"}, rows)
	return nil
}
