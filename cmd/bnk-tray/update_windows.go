//go:build windows

package main

import (
	"os"
	"path/filepath"

	"golang.org/x/sys/windows"

	"github.com/christ-pher/bnk/internal/selfupdate"
)

// msiAsset is the installer's name in a release, which is also its key
// in the release's SHA256SUMS.
const msiAsset = "bnk-windows-amd64.msi"

// launchUpdater downloads the release's MSI, checks it against the
// release's SHA256SUMS, and hands it to Windows to install. The check
// gives the installer the same verification the Linux binary path has
// always had: what runs is what the release workflow checksummed, not
// merely whatever the download returned. Installing per-machine needs
// elevation, so msiexec raises the consent prompt itself — the tray
// never holds privileges it would then have to be trusted with.
func launchUpdater(tag string) error {
	msi, err := selfupdate.FetchVerified(repoURL, tag, msiAsset)
	if err != nil {
		return err
	}
	dst := filepath.Join(os.TempDir(), "bnk-"+tag+".msi")
	if err := os.WriteFile(dst, msi, 0o600); err != nil {
		return err
	}
	return shellExecute("open", dst, "")
}

// shellExecute runs a file the way a double-click would, which is what
// triggers the installer's own elevation prompt.
func shellExecute(verb, file, args string) error {
	v, err := windows.UTF16PtrFromString(verb)
	if err != nil {
		return err
	}
	f, err := windows.UTF16PtrFromString(file)
	if err != nil {
		return err
	}
	var a *uint16
	if args != "" {
		if a, err = windows.UTF16PtrFromString(args); err != nil {
			return err
		}
	}
	return windows.ShellExecute(0, v, f, a, nil, windows.SW_SHOWNORMAL)
}
