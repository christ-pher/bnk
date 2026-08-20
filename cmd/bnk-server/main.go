// bnk-server is the control server: it enrolls nodes, assigns IPs, pushes
// netmaps, and (soon) relays traffic. Admin subcommands talk to a running
// bnk-server over a unix socket in the state directory.
package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/christ-pher/bnk/internal/cliutil"
	"github.com/christ-pher/bnk/internal/coord/server"
	"github.com/christ-pher/bnk/internal/pin"
	"github.com/christ-pher/bnk/internal/selfupdate"
	"github.com/christ-pher/bnk/internal/store"
	"github.com/christ-pher/bnk/internal/stunner"
)

// version is stamped by the release workflow (-X main.version=vX.Y.Z);
// local builds report "dev".
var version = "dev"

// rawBase is where the install scripts are served from.
const rawBase = "https://raw.githubusercontent.com/christ-pher/bnk/main"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "bnk-server:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: bnk-server <up|down|status|serve|key|node|net|acl|update|version> [flags]")
	}
	switch args[0] {
	case "serve":
		return serve(args[1:])
	case "up":
		return systemctl("start")
	case "down":
		return systemctl("stop")
	case "status":
		return statusCmd(args[1:])
	case "net":
		if len(args) < 2 {
			return fmt.Errorf("usage: bnk-server net <get|set> [cidr]")
		}
		switch args[1] {
		case "get":
			return netGet(args[2:])
		case "set":
			return netSet(args[2:])
		default:
			return fmt.Errorf("usage: bnk-server net <get|set> [cidr]")
		}
	case "version":
		fmt.Println(version)
		return nil
	case "update":
		if os.Geteuid() != 0 {
			return fmt.Errorf("update replaces /usr/local/bin/bnk-server — run as root (sudo bnk-server update)")
		}
		return selfupdate.Run(selfupdate.Config{
			BaseURL: "https://github.com/christ-pher/bnk",
			Asset:   "bnk-server",
			Version: version,
			Service: "bnk-server",
		})
	case "key":
		if len(args) < 2 {
			return fmt.Errorf("usage: bnk-server key <new|ls|revoke> ...")
		}
		switch args[1] {
		case "new":
			return keyNew(args[2:])
		case "ls":
			return keyLs(args[2:])
		case "revoke":
			return keyRevoke(args[2:])
		default:
			return fmt.Errorf("usage: bnk-server key <new|ls|revoke> ...")
		}
	case "node":
		if len(args) < 2 || args[1] != "ls" {
			return fmt.Errorf("usage: bnk-server node ls")
		}
		return nodeLs(args[2:])
	case "acl":
		if len(args) < 2 {
			return fmt.Errorf("usage: bnk-server acl <set|get|check> ...")
		}
		switch args[1] {
		case "set":
			return aclSet(args[2:])
		case "get":
			return aclGet(args[2:])
		case "check":
			return aclCheck(args[2:])
		default:
			return fmt.Errorf("usage: bnk-server acl <set|get|check> ...")
		}
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func stateDirFlag(fs *flag.FlagSet) *string {
	return fs.String("state-dir", "/var/lib/bnk-server", "directory for state, cert, and admin socket")
}

func netGet(args []string) error {
	fs := flag.NewFlagSet("net get", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)
	resp, err := adminClient(*stateDir).Get("http://bnk-server/network")
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
	}
	defer resp.Body.Close()
	var out struct {
		Network string `json:"network"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	fmt.Println(out.Network)
	return nil
}

// netSet moves the whole mesh to a different network. Every node is
// re-addressed and reconnects on its own; the operation interrupts
// traffic, so it confirms first unless --yes is given.
func netSet(args []string) error {
	fs := flag.NewFlagSet("net set", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	yes := fs.Bool("yes", false, "skip the confirmation prompt")
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: bnk-server net set <cidr>   e.g. bnk-server net set 100.67.0.0/16")
	}
	target := fs.Arg(0)
	if _, err := netip.ParsePrefix(target); err != nil {
		return fmt.Errorf("bad network %q: %w", target, err)
	}

	if !*yes {
		fmt.Printf("Move the mesh to %s?\n", target)
		fmt.Println("Every node is re-addressed and its tunnel restarts; traffic drops briefly.")
		fmt.Print("Type yes to continue: ")
		var answer string
		fmt.Scanln(&answer)
		if answer != "yes" {
			fmt.Println("cancelled")
			return nil
		}
	}

	body, err := json.Marshal(map[string]string{"network": target})
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPut, "http://bnk-server/network", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := adminClient(*stateDir).Do(req)
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s", bytes.TrimSpace(msg))
	}
	fmt.Printf("mesh network is now %s\n", target)
	fmt.Println("Connected nodes re-address themselves; offline nodes do it when they reconnect.")
	fmt.Println("Check with: bnk-server node ls")
	return nil
}

// statusCmd reports whether the control server is running, via its admin
// socket, with a node summary when it is.
func statusCmd(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)

	hc := adminClient(*stateDir)
	resp, err := hc.Get("http://bnk-server/info")
	if err != nil {
		if errors.Is(err, os.ErrPermission) {
			return fmt.Errorf("the admin socket needs root: sudo bnk-server status")
		}
		fmt.Println("bnk-server is down (run `bnk-server up` to start it)")
		return nil
	}
	var info struct {
		PublicURL string `json:"public_url"`
		Network   string `json:"network"`
	}
	err = json.NewDecoder(resp.Body).Decode(&info)
	resp.Body.Close()
	if err != nil {
		return err
	}

	line := "bnk-server is up"
	if info.PublicURL != "" {
		line += ": " + info.PublicURL
	}
	if info.Network != "" {
		line += "  network " + info.Network
	}
	nresp, err := hc.Get("http://bnk-server/nodes")
	if err == nil {
		var nodes []server.AdminNode
		derr := json.NewDecoder(nresp.Body).Decode(&nodes)
		nresp.Body.Close()
		if derr == nil {
			online := 0
			for _, n := range nodes {
				if n.Online {
					online++
				}
			}
			line += fmt.Sprintf(" — %d nodes (%d online)", len(nodes), online)
		}
	}
	fmt.Println(line)
	return nil
}

// systemctl starts or stops the bnk-server service: `bnk-server up` / `bnk-server down`
// control the control server without remembering systemd incantations.
func systemctl(verb string) error {
	cmd := exec.Command("systemctl", verb, "bnk-server")
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("systemctl %s bnk-server: %w (is the bnk-server service installed? see install-server.sh)", verb, err)
	}
	if verb == "start" {
		fmt.Println("bnk-server is up")
	} else {
		fmt.Println("bnk-server is down (run `bnk-server up` to start it again)")
	}
	return nil
}

func serve(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	listen := fs.String("listen", ":8443", "TLS listen address for clients")
	publicURL := fs.String("public-url", "", "URL clients reach this server at (default: detected outbound IP + listen port)")
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
	pubURL := *publicURL
	if pubURL == "" {
		pubURL = detectPublicURL(*listen)
	}
	go http.Serve(adminLn, srv.AdminHandler(fp, pubURL))

	fmt.Printf("bnk-server: listening on %s\n", *listen)
	fmt.Printf("bnk-server: cert fingerprint %s\n", fp)
	fmt.Printf("bnk-server: mint enrollment keys with: bnk-server key new --state-dir %s\n", *stateDir)

	hs := &http.Server{
		Addr:      *listen,
		Handler:   srv.Handler(),
		TLSConfig: &tls.Config{Certificates: []tls.Certificate{cert}},
	}
	return hs.ListenAndServeTLS("", "")
}

// detectPublicURL guesses the URL clients can reach this server at: the
// machine's outbound IP plus the listen port. No packets are sent — the
// UDP dial only resolves a route. A server behind NAT should pass
// --public-url instead.
func detectPublicURL(listen string) string {
	_, port, err := net.SplitHostPort(listen)
	if err != nil {
		return ""
	}
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return ""
	}
	defer conn.Close()
	ua, ok := conn.LocalAddr().(*net.UDPAddr)
	if !ok {
		return ""
	}
	return "https://" + net.JoinHostPort(ua.IP.String(), port)
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

	u := fmt.Sprintf("http://bnk-server/enroll-keys?ttl=%s&reusable=%v", url.QueryEscape(ttl.String()), *reusable)
	resp, err := adminClient(*stateDir).Post(u, "application/json", nil)
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
	}
	defer resp.Body.Close()
	var out struct {
		Key string `json:"key"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	fmt.Println(out.Key)
	printInstallHint(*stateDir, out.Key)
	return nil
}

// printInstallHint writes a ready-to-paste client install command to
// stderr — stderr so `KEY=$(bnk-server key new)` still captures only the
// key on stdout.
func printInstallHint(stateDir, key string) {
	serverURL := "https://YOUR_SERVER_IP:8443"
	var info struct {
		PublicURL string `json:"public_url"`
		Network   string `json:"network"`
	}
	if resp, err := adminClient(stateDir).Get("http://bnk-server/info"); err == nil {
		json.NewDecoder(resp.Body).Decode(&info)
		resp.Body.Close()
		if info.PublicURL != "" {
			serverURL = info.PublicURL
		}
	}
	// Both platforms are printed because the server cannot know what the
	// joining machine runs. Everything here goes to stderr so stdout
	// stays a bare key for scripts.
	const rule = "────────────────────────────────────────────────────────────"
	w := os.Stderr
	fmt.Fprintf(w, "\n%s\n", rule)
	fmt.Fprintf(w, "  Join a node — paste ONE of these on that machine.\n")
	fmt.Fprintf(w, "  Server: %s   (check this is reachable from the node)\n", serverURL)
	fmt.Fprintf(w, "%s\n\n", rule)
	// Each command stays on one line: a wrapped line survives a
	// triple-click copy, a backslash continuation does not.
	fmt.Fprintf(w, "  LINUX  (as root)\n\n")
	fmt.Fprintf(w, "    curl -fsSL %s/install-client.sh | sudo sh -s -- --server %s --key %s\n\n",
		rawBase, serverURL, key)
	fmt.Fprintf(w, "  WINDOWS  (elevated PowerShell)\n\n")
	fmt.Fprintf(w, "    & ([scriptblock]::Create((irm %s/install-client.ps1))) -Server %s -Key %s\n\n",
		rawBase, serverURL, key)
	fmt.Fprintf(w, "  The key is single-use and expires — mint another with `bnk-server key new`.\n")
	fmt.Fprintf(w, "%s\n", rule)
}

func aclSet(args []string) error {
	fs := flag.NewFlagSet("acl set", flag.ExitOnError)
	stateDir := stateDirFlag(fs)
	fs.Parse(args)
	if fs.NArg() != 1 {
		return fmt.Errorf("usage: bnk-server acl set <policy.json>")
	}
	policy, err := os.ReadFile(fs.Arg(0))
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPut, "http://bnk-server/policy", bytes.NewReader(policy))
	if err != nil {
		return err
	}
	resp, err := adminClient(*stateDir).Do(req)
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
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
	resp, err := adminClient(*stateDir).Get("http://bnk-server/policy")
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
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
		return fmt.Errorf("usage: bnk-server acl check <src-node> <dst-node> <tcp/22|udp/53|icmp>")
	}
	u := fmt.Sprintf("http://bnk-server/check?src=%s&dst=%s&target=%s",
		url.QueryEscape(fs.Arg(0)), url.QueryEscape(fs.Arg(1)), url.QueryEscape(fs.Arg(2)))
	resp, err := adminClient(*stateDir).Get(u)
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
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

	resp, err := adminClient(*stateDir).Get("http://bnk-server/enroll-keys")
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
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
		return fmt.Errorf("usage: bnk-server key revoke <prefix>")
	}
	resp, err := adminClient(*stateDir).Post("http://bnk-server/enroll-keys/revoke?prefix="+url.QueryEscape(fs.Arg(0)), "application/json", nil)
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
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

	resp, err := adminClient(*stateDir).Get("http://bnk-server/nodes")
	if err != nil {
		return fmt.Errorf("is bnk-server running? %w", err)
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
