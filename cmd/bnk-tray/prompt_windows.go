//go:build windows

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/sys/windows"
)

// promptScript shows a two-field-worth prompt in one box. It is written
// to a file rather than passed with -Command: a multi-line script given
// to -Command is mis-parsed, which is why the box never appeared.
const promptScript = `Add-Type -AssemblyName Microsoft.VisualBasic
$msg = "Paste the join command from your server.` + "`n`n" + `Run 'bnk-server key new' on the server and paste the whole Windows line, or just the bnkkey:... value."
$r = [Microsoft.VisualBasic.Interaction]::InputBox($msg, 'Sign in to bnk', '')
[Console]::Out.Write($r)
`

// promptForJoin asks for a join command and reports why it could not,
// because a prompt that silently does nothing is worse than an error.
func promptForJoin() (string, error) {
	s, err := inputBox()
	if err == nil {
		if s == "" {
			return "", errCancelled
		}
		return s, nil
	}

	// The dialog failed. The clipboard is a real fallback here: the join
	// command is something the user copied from the server anyway.
	if clip, ok := getClipboard(); ok {
		return clip, nil
	}
	return "", fmt.Errorf("could not open the sign-in box (%w), and the clipboard is empty — copy the join command from `bnk-server key new`, then try again", err)
}

// errCancelled marks the user closing the box, which is not a failure.
var errCancelled = fmt.Errorf("cancelled")

func inputBox() (string, error) {
	dir, err := os.MkdirTemp("", "bnk-prompt")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "prompt.ps1")
	if err := os.WriteFile(path, []byte(promptScript), 0o600); err != nil {
		return "", err
	}

	// -File avoids the quoting and newline handling that breaks
	// -Command; ExecutionPolicy Bypass is needed because the default
	// policy refuses to run a script file at all.
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
		"-STA", "-WindowStyle", "Hidden", "-File", path)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			return "", fmt.Errorf("%v: %s", err, strings.TrimSpace(string(ee.Stderr)))
		}
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// alert shows a message the user cannot miss, unlike a line in a menu
// they have already closed.
func alert(text, caption string) {
	t, err := windows.UTF16PtrFromString(text)
	if err != nil {
		return
	}
	c, err := windows.UTF16PtrFromString(caption)
	if err != nil {
		return
	}
	// MB_OK | MB_SETFOREGROUND | MB_TOPMOST
	windows.MessageBox(0, t, c, 0x0|0x10000|0x40000)
}
