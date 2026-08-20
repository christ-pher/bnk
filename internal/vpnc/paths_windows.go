//go:build windows

package vpnc

import (
	"os"
	"path/filepath"
)

// DefaultSocket is the diagnostics named pipe; the control pipe is
// ControlPipe(DefaultSocket). Named pipes carry their own ACLs, which is
// how Windows enforces the split that Linux enforces with SO_PEERCRED.
const DefaultSocket = `\\.\pipe\bnk`

// DefaultStateDir holds the node's identity under ProgramData. The
// drive is read from the environment rather than assumed to be C:.
var DefaultStateDir = defaultStateDir()

func defaultStateDir() string {
	if pd := os.Getenv("ProgramData"); pd != "" {
		return filepath.Join(pd, "bnk")
	}
	return `C:\ProgramData\bnk`
}
