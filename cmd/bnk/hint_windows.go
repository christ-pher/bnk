//go:build windows

package main

import "fmt"

// adminHint explains the two common Windows causes: no Administrator
// rights, or wintun.dll missing from the executable's directory.
func adminHint(err error) error {
	return fmt.Errorf("needs Administrator, and wintun.dll must sit next to bnk.exe: %w", err)
}
