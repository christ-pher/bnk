//go:build windows

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

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
	fs.Parse(args)
	if *server == "" {
		return fmt.Errorf("--server is required")
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

	binPath := commandLine(exe, serviceArgs(*server, *key, *stateDir))

	// Re-running the installer must update the existing service rather
	// than fail — that is also how a spent enrollment key gets dropped.
	if s, err := m.OpenService(serviceName); err == nil {
		defer s.Close()
		cfg, err := s.Config()
		if err != nil {
			return err
		}
		cfg.BinaryPathName = binPath
		cfg.StartType = mgr.StartAutomatic
		if err := s.UpdateConfig(cfg); err != nil {
			return err
		}
		fmt.Println("bnk service updated")
		return nil
	}

	s, err := m.CreateService(serviceName, exe, mgr.Config{
		DisplayName:    "bnk mesh VPN",
		Description:    "Connects this machine to the bnk mesh network.",
		StartType:      mgr.StartAutomatic,
		BinaryPathName: binPath,
	})
	if err != nil {
		return err
	}
	defer s.Close()
	fmt.Println("bnk service installed")
	return nil
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
	if err := s.Delete(); err != nil {
		return err
	}
	fmt.Println("bnk service removed")
	return nil
}

// commandLine quotes the executable path so a service ImagePath with
// spaces (C:\Program Files\bnk\bnk.exe) parses correctly.
func commandLine(exe string, args []string) string {
	return `"` + exe + `" ` + strings.Join(args, " ")
}

// updateCmd: Windows releases ship as a zip and swapping a running .exe
// needs rename-and-restart handling, so the installer is the update path.
func updateCmd() error {
	return fmt.Errorf("on Windows, update by re-running the installer:\n" +
		`  & ([scriptblock]::Create((irm ` + rawInstallerURL + `))) -Server <your server URL>`)
}
