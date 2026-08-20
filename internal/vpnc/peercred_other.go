//go:build !linux

package vpnc

import "net"

// peerUID is unavailable off Linux; callers treat unknown as unprivileged.
func peerUID(net.Conn) (int, bool) {
	return 0, false
}
