//go:build windows

package main

import (
	"os/exec"
	"strings"
	"syscall"
)

// promptForJoin shows a text box to paste a join command into.
//
// It shells out to PowerShell's InputBox rather than building a dialog:
// systray has no text input, and a hand-rolled Win32 dialog would be
// hundreds of lines of untestable code for one rarely-used prompt.
func promptForJoin() (string, bool) {
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
