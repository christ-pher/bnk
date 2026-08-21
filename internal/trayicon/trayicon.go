// Package trayicon holds the tray's status icons.
//
// They live outside cmd/bnk-tray because that binary is Windows-only:
// embedded here, the artwork is checked on every platform's test run
// rather than only when someone builds for Windows.
//
// The images are generated from the logo by packaging/icons/gen.py.
// Edit that script, not these files.
package trayicon

import _ "embed"

// Connected is shown while the tunnel is up.
//
//go:embed icons/connected.ico
var Connected []byte

// Disconnected is shown when the tunnel is down on purpose.
//
//go:embed icons/disconnected.ico
var Disconnected []byte

// Attention is shown when the tray cannot do its job unaided: the
// daemon is not running, or this machine has not signed in. Both are
// states the user has to act on, and grey read as "merely off".
//
//go:embed icons/attention.ico
var Attention []byte
