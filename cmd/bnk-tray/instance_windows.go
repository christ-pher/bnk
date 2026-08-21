//go:build windows

package main

import (
	"golang.org/x/sys/windows"
)

// claimSingleInstance takes a named mutex for this logon session. The
// name is session-scoped, not Global, so two users can each run their
// own tray. The returned func releases it.
func claimSingleInstance() (release func(), alreadyRunning bool) {
	name, err := windows.UTF16PtrFromString(`Local\bnk-tray`)
	if err != nil {
		return func() {}, false
	}
	h, err := windows.CreateMutex(nil, false, name)
	if err == windows.ERROR_ALREADY_EXISTS {
		if h != 0 {
			windows.CloseHandle(h)
		}
		return func() {}, true
	}
	if err != nil && h == 0 {
		// Cannot tell; better to run than to refuse to start.
		return func() {}, false
	}
	return func() { windows.CloseHandle(h) }, false
}
