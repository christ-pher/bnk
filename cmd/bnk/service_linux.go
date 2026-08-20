//go:build linux

package main

import (
	"fmt"
	"os"

	"github.com/christ-pher/bnk/internal/selfupdate"
)

// serviceCmd exists so `bnk service ...` gives the same answer on both
// platforms; on Linux the unit is managed by the install script.
func serviceCmd(args []string) error {
	return fmt.Errorf("on Linux the service is a systemd unit — install or remove it with install-client.sh (see DEPLOY.md)")
}

func updateCmd() error {
	if os.Geteuid() != 0 {
		return fmt.Errorf("update replaces /usr/local/bin/bnk — run as root (sudo bnk update)")
	}
	return selfupdate.Run(selfupdate.Config{
		BaseURL: repoURL,
		Asset:   "bnk",
		Version: version,
		Service: "bnk",
	})
}
