//go:build windows

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"golang.org/x/sys/windows"
)

// msiURL is where a release publishes its installer.
func msiURL(tag string) string {
	return fmt.Sprintf("%s/releases/download/%s/bnk-windows-amd64.msi", repoURL, tag)
}

// launchUpdater downloads the release's MSI and hands it to Windows to
// install. Installing per-machine needs elevation, so msiexec raises the
// consent prompt itself — the tray never holds privileges it would then
// have to be trusted with.
func launchUpdater(tag string) error {
	dst := filepath.Join(os.TempDir(), "bnk-"+tag+".msi")
	if err := download(msiURL(tag), dst); err != nil {
		return err
	}
	return shellExecute("open", dst, "")
}

func download(url, dst string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
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
