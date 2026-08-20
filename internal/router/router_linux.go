//go:build linux

// Package router applies OS network configuration for the tunnel interface.
package router

import (
	"fmt"
	"net/netip"
	"os/exec"
	"strconv"

	"golang.zx2c4.com/wireguard/tun"
)

// Up assigns prefix (the node's IP with the mesh prefix length, so the
// kernel derives the mesh route) and brings the interface up. dev is
// unused here — Linux addresses the interface by name — but Windows
// needs it for the adapter LUID.
func Up(dev tun.Device, ifName string, prefix netip.Prefix, mtu int) error {
	cmds := [][]string{
		{"ip", "addr", "add", prefix.String(), "dev", ifName},
		{"ip", "link", "set", ifName, "up", "mtu", strconv.Itoa(mtu)},
	}
	for _, c := range cmds {
		if out, err := exec.Command(c[0], c[1:]...).CombinedOutput(); err != nil {
			return fmt.Errorf("router: %v: %v: %s", c, err, out)
		}
	}
	return nil
}
