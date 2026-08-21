//go:build windows

package vpnc

import (
	"fmt"

	"golang.org/x/sys/windows"
)

// hardenStateDir restricts the state directory to SYSTEM and
// Administrators.
//
// This exists because Go's file modes are silently meaningless on
// Windows: os.MkdirAll's perm bits are ignored and os.Chmod only toggles
// the read-only attribute. %ProgramData% grants BUILTIN\Users read by
// inheritance, so client.json — which holds the node's private key —
// was readable by every local account, including malware running as an
// unprivileged user.
func hardenStateDir(dir string) error {
	// Protected (P) blocks inheritance from ProgramData; without it the
	// Users ACE would still apply.
	sd, err := windows.SecurityDescriptorFromString("D:P(A;OICI;GA;;;SY)(A;OICI;GA;;;BA)")
	if err != nil {
		return err
	}
	dacl, _, err := sd.DACL()
	if err != nil {
		return err
	}
	if err := windows.SetNamedSecurityInfo(
		dir,
		windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION,
		nil, nil, dacl, nil,
	); err != nil {
		return fmt.Errorf("restricting %s: %w", dir, err)
	}
	return nil
}
