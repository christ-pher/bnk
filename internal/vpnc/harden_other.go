//go:build !windows

package vpnc

// hardenStateDir is a no-op away from Windows: the state directory is
// created 0700 and the state file 0600, which the OS actually honours.
func hardenStateDir(string) error { return nil }
