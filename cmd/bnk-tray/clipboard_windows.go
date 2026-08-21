//go:build windows

package main

import (
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Clipboard access through the Win32 API directly: the alternative is a
// dependency, and this is one call sequence.
var (
	user32             = windows.NewLazySystemDLL("user32.dll")
	procOpenClipboard  = user32.NewProc("OpenClipboard")
	procCloseClipboard = user32.NewProc("CloseClipboard")
	procEmptyClipboard = user32.NewProc("EmptyClipboard")
	procSetClipboard   = user32.NewProc("SetClipboardData")

	kernel32         = windows.NewLazySystemDLL("kernel32.dll")
	procGlobalAlloc  = kernel32.NewProc("GlobalAlloc")
	procGlobalLock   = kernel32.NewProc("GlobalLock")
	procGlobalUnlock = kernel32.NewProc("GlobalUnlock")
	procGlobalFree   = kernel32.NewProc("GlobalFree")
	procGlobalSize   = kernel32.NewProc("GlobalSize")
	procGetClipboard = user32.NewProc("GetClipboardData")
	procMoveMemory   = kernel32.NewProc("RtlMoveMemory")
)

const (
	cfUnicodeText = 13
	gmemMoveable  = 0x0002
)

// setClipboard puts s on the clipboard as unicode text. Failures are
// silent by design: a tray click that cannot copy is not worth an error
// dialog, and the address is already visible in the menu.
func setClipboard(s string) {
	utf16, err := windows.UTF16FromString(s)
	if err != nil {
		return
	}
	if r, _, _ := procOpenClipboard.Call(0); r == 0 {
		return
	}
	defer procCloseClipboard.Call()
	procEmptyClipboard.Call()

	size := uintptr(len(utf16) * 2)
	h, _, _ := procGlobalAlloc.Call(gmemMoveable, size)
	if h == 0 {
		return
	}
	ptr, _, _ := procGlobalLock.Call(h)
	if ptr == 0 {
		procGlobalFree.Call(h)
		return
	}
	// Copy through RtlMoveMemory rather than reinterpreting the locked
	// address as a Go slice: this memory belongs to Windows, and taking a
	// Go pointer to it is exactly what unsafe.Pointer rules forbid.
	procMoveMemory.Call(ptr, uintptr(unsafe.Pointer(&utf16[0])), size)
	procGlobalUnlock.Call(h)

	// On success the clipboard owns the memory; on failure we still do.
	if r, _, _ := procSetClipboard.Call(cfUnicodeText, h); r == 0 {
		procGlobalFree.Call(h)
	}
}

// getClipboard reads unicode text from the clipboard, the fallback the
// sign-in prompt uses when a dialog cannot be shown.
func getClipboard() (string, bool) {
	if r, _, _ := procOpenClipboard.Call(0); r == 0 {
		return "", false
	}
	defer procCloseClipboard.Call()

	h, _, _ := procGetClipboard.Call(cfUnicodeText)
	if h == 0 {
		return "", false
	}
	ptr, _, _ := procGlobalLock.Call(h)
	if ptr == 0 {
		return "", false
	}
	defer procGlobalUnlock.Call(h)

	size, _, _ := procGlobalSize.Call(h)
	if size < 2 {
		return "", false
	}
	// Copy out through RtlMoveMemory for the same reason the write path
	// does: this memory belongs to Windows, not to Go.
	buf := make([]uint16, size/2)
	procMoveMemory.Call(uintptr(unsafe.Pointer(&buf[0])), ptr, size)
	s := strings.TrimSpace(windows.UTF16ToString(buf))
	return s, s != ""
}
