//go:build windows

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"syscall"

	"time"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/svc"
	"golang.org/x/sys/windows/svc/mgr"

	"github.com/christ-pher/bnk/internal/vpnc"
)

// serviceCmd registers or removes the Windows service. The PowerShell
// installer calls it so the logic lives here rather than in the script.
func serviceCmd(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: bnk service <install|uninstall> [flags]")
	}
	switch args[0] {
	case "install":
		return serviceInstall(args[1:])
	case "uninstall":
		return serviceUninstall()
	default:
		return fmt.Errorf("usage: bnk service <install|uninstall> [flags]")
	}
}

func serviceInstall(args []string) error {
	fs := flag.NewFlagSet("service install", flag.ExitOnError)
	server := fs.String("server", "", "control server URL, e.g. https://host:8443")
	key := fs.String("key", "", "enrollment key (bnkkey:...); omit once enrolled")
	stateDir := fs.String("state-dir", vpnc.DefaultStateDir, "directory for client state")
	operator := fs.String("operator", "", "account (SID or name) allowed to toggle the tunnel without elevating")
	start := fs.Bool("start", false, "start the service once it is registered")
	fs.Parse(args)
	// No --server is fine: the daemon idles unenrolled and waits to be
	// signed in from the tray. Requiring one here is what left a
	// double-clicked install with no service at all.
	// Accept either a SID or an account name: the MSI can only pass a
	// name, since Windows Installer has no built-in SID property.
	operatorSID, err := resolveOperator(*operator)
	if err != nil {
		return err
	}

	exe, err := os.Executable()
	if err != nil {
		return err
	}
	m, err := mgr.Connect()
	if err != nil {
		return fmt.Errorf("connect to service manager (run as Administrator): %w", err)
	}
	defer m.Disconnect()

	svcArgs := serviceArgs(*server, *key, *stateDir, operatorSID)

	// Re-running the installer must update the existing service rather
	// than fail — that is also how a spent enrollment key gets dropped.
	if s, err := m.OpenService(serviceName); err == nil {
		defer s.Close()
		cfg, err := s.Config()
		if err != nil {
			return err
		}
		// UpdateConfig, unlike CreateService, takes the whole command
		// line through BinaryPathName.
		cfg.BinaryPathName = commandLine(exe, svcArgs)
		cfg.StartType = mgr.StartAutomatic
		if err := s.UpdateConfig(cfg); err != nil {
			return err
		}
		setTrayAutostart(operatorSID, exe)
		fmt.Println("bnk service updated")
		if *start {
			return startService()
		}
		return nil
	}

	// The arguments MUST be passed variadically: CreateService builds the
	// registered command line from exe plus these, and ignores
	// Config.BinaryPathName. Setting that field instead registers a bare
	// exe with no arguments, which starts, prints usage, and exits 1.
	s, err := m.CreateService(serviceName, exe, mgr.Config{
		DisplayName: "bnk mesh VPN",
		Description: "Connects this machine to the bnk mesh network.",
		StartType:   mgr.StartAutomatic,
	}, svcArgs...)
	if err != nil {
		return err
	}
	defer s.Close()
	setTrayAutostart(operatorSID, exe)
	fmt.Println("bnk service installed")
	if *start {
		return startService()
	}
	return nil
}

// startService starts the registered service, treating "already
// running" as success so a re-install is idempotent.
func startService() error {
	m, err := mgr.Connect()
	if err != nil {
		return err
	}
	defer m.Disconnect()
	s, err := m.OpenService(serviceName)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := s.Start(); err != nil {
		if st, serr := s.Query(); serr == nil && st.State == svc.Running {
			return nil
		}
		return fmt.Errorf("start bnk service: %w", err)
	}
	return nil
}

// resolveOperator turns an account name into a SID, and passes an
// existing SID through untouched. An empty operator stays empty, which
// means administrators only.
func resolveOperator(operator string) (string, error) {
	if operator == "" {
		return "", nil
	}
	if sidPattern.MatchString(operator) {
		return operator, nil
	}
	sid, _, _, err := windows.LookupSID("", operator)
	if err != nil {
		return "", fmt.Errorf("cannot resolve operator %q to an account: %w", operator, err)
	}
	return sid.String(), nil
}

// sidPattern mirrors the daemon's own check so a name that merely looks
// like a SID is not silently treated as one.
var sidPattern = regexp.MustCompile(`^S-1-\d+(-\d+)+$`)

// setTrayAutostart is best-effort: a machine with no tray installed, or
// an operator whose hive is not loaded, must not fail the install.
func setTrayAutostart(operatorSID, exe string) {
	if operatorSID == "" {
		return
	}
	if err := enableTrayAutostart(operatorSID, filepath.Dir(exe)); err != nil {
		fmt.Fprintf(os.Stderr, "note: could not set the tray to start at login: %v\n", err)
	}
}

func serviceUninstall() error {
	m, err := mgr.Connect()
	if err != nil {
		return fmt.Errorf("connect to service manager (run as Administrator): %w", err)
	}
	defer m.Disconnect()
	s, err := m.OpenService(serviceName)
	if err != nil {
		return fmt.Errorf("bnk service is not installed")
	}
	defer s.Close()
	// The tray holds its own executable open, so an uninstaller that
	// only stops the service still cannot delete the files.
	stopTray()

	// Clear the tray's autostart entry for whichever operator the
	// registered command line names, before the service record is gone.
	if cfg, err := s.Config(); err == nil {
		if sid := operatorFromCommandLine(cfg.BinaryPathName); sid != "" {
			disableTrayAutostart(sid)
		}
	}
	stopService(s)
	if err := s.Delete(); err != nil {
		return err
	}
	fmt.Println("bnk service removed")
	return nil
}

// commandLine renders a service ImagePath, escaping each element the
// same way CreateService does internally so the create and update paths
// register byte-identical command lines.
func commandLine(exe string, args []string) string {
	s := syscall.EscapeArg(exe)
	for _, a := range args {
		s += " " + syscall.EscapeArg(a)
	}
	return s
}

// updateCmd: Windows releases ship as a zip and swapping a running .exe
// needs rename-and-restart handling, so the installer is the update path.
func updateCmd() error {
	return fmt.Errorf("on Windows, update by re-running the installer:\n" +
		`  & ([scriptblock]::Create((irm ` + rawInstallerURL + `))) -Server <your server URL>`)
}

// stopTray ends any running tray process. It is best-effort: no tray
// running is the normal case, and taskkill reports that as a failure.
func stopTray() {
	cmd := exec.Command("taskkill", "/IM", "bnk-tray.exe", "/F")
	cmd.Run()
}

// stopService stops the service and waits for it, because deleting a
// running service only marks it for deletion — leaving its process
// holding the very files an uninstall is about to remove.
func stopService(s *mgr.Service) {
	st, err := s.Control(svc.Stop)
	if err != nil {
		return // not running, or already stopping
	}
	deadline := time.Now().Add(20 * time.Second)
	for st.State != svc.Stopped && time.Now().Before(deadline) {
		time.Sleep(300 * time.Millisecond)
		if st, err = s.Query(); err != nil {
			return
		}
	}
}
