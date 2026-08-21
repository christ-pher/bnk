//go:build windows

package main

import (
	"os/exec"
	"strings"
	"syscall"
)

// promptForJoin asks for a join command.
//
// It shells out to PowerShell's InputBox rather than building a dialog:
// systray has no text input, and a hand-rolled Win32 dialog would be
// hundreds of lines of untestable code for one rarely-used prompt. If
// that fails for any reason — a locked-down PowerShell, a missing
// assembly — the clipboard is used instead, which works because the
// server prints a join command people copy anyway.
func promptForJoin() (string, bool) {
	if s, ok := inputBox(); ok {
		return s, true
	}
	if s, ok := getClipboard(); ok {
		return s, true
	}
	return "", false
}

func inputBox() (string, bool) {
	const script = `Add-Type -AssemblyName Microsoft.VisualBasic;
[Microsoft.VisualBasic.Interaction]::InputBox(
  'Paste the join command from your server (bnk-server key new), or just the bnkkey:... line.',
  'Sign in to bnk', '')`

	cmd := exec.Command("powershell", "-NoProfile", "-STA", "-Command", script)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	s := strings.TrimSpace(string(out))
	return s, s != ""
}
