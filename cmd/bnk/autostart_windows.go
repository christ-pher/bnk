//go:build windows

package main

import (
	"fmt"
	"path/filepath"

	"golang.org/x/sys/windows/registry"
)

// The tray runs as the operator, not as the service account, so its
// autostart entry belongs in that user's hive. Writing it by SID rather
// than through HKCU matters because whoever registers the service is
// elevated — and may be SYSTEM, under an MSI — so HKCU is the wrong
// hive at exactly the moment we need to write it.
const runKeyPath = `Software\Microsoft\Windows\CurrentVersion\Run`

const trayValueName = "bnk-tray"

// enableTrayAutostart points the operator's Run key at bnk-tray.exe,
// which is installed next to bnk.exe.
func enableTrayAutostart(operatorSID, exeDir string) error {
	if operatorSID == "" {
		return nil // no operator: nobody to start a tray for
	}
	k, err := registry.OpenKey(registry.USERS, operatorSID+`\`+runKeyPath, registry.SET_VALUE)
	if err != nil {
		return fmt.Errorf("open Run key for %s (is that user logged in?): %w", operatorSID, err)
	}
	defer k.Close()
	tray := filepath.Join(exeDir, "bnk-tray.exe")
	return k.SetStringValue(trayValueName, `"`+tray+`"`)
}

// disableTrayAutostart removes the entry; a missing value is success.
func disableTrayAutostart(operatorSID string) error {
	if operatorSID == "" {
		return nil
	}
	k, err := registry.OpenKey(registry.USERS, operatorSID+`\`+runKeyPath, registry.SET_VALUE)
	if err != nil {
		return nil
	}
	defer k.Close()
	if err := k.DeleteValue(trayValueName); err != nil && err != registry.ErrNotExist {
		return err
	}
	return nil
}
