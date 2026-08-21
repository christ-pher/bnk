//go:build windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// noArgs: on Windows the tray is the interface people actually mean, so
// a bare `bnk` opens it rather than printing a usage block at someone
// who is looking for a window.
func noArgs() error {
	fmt.Println("opening the bnk tray (run `bnk help` for command-line usage)")
	return trayCmd()
}

// trayCmd launches the tray beside this executable and returns without
// waiting, so the console does not stay tied to it.
func trayCmd() error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	tray := filepath.Join(filepath.Dir(exe), "bnk-tray.exe")
	if _, err := os.Stat(tray); err != nil {
		return fmt.Errorf("bnk-tray.exe is not installed next to bnk.exe (%s)", tray)
	}
	cmd := exec.Command(tray)
	if err := cmd.Start(); err != nil {
		return err
	}
	return cmd.Process.Release()
}
