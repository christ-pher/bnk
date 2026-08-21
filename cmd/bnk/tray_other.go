//go:build !windows

package main

import "fmt"

func noArgs() error {
	return fmt.Errorf("usage: bnk <up|down|status|join|leave|ping|netcheck|update|version|run|service> [flags]")
}

func trayCmd() error {
	return fmt.Errorf("the tray is Windows-only; use `bnk status`, `bnk up`, and `bnk down`")
}
