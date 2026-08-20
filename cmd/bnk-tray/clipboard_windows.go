//go:build windows

package main

import (
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
