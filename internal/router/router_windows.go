//go:build windows

// Package router applies OS network configuration for the tunnel interface.
package router

import (
	"fmt"
	"net/netip"

	"golang.zx2c4.com/wireguard/tun"
	"golang.zx2c4.com/wireguard/windows/tunnel/winipcfg"
)

// Up assigns prefix to the Wintun adapter and sets its MTU. Unlike
// Linux there is no command to shell out to: the adapter is addressed by
// its LUID, which only the device object knows.
func Up(dev tun.Device, ifName string, prefix netip.Prefix, mtu int) error {
	nt, ok := dev.(*tun.NativeTun)
	if !ok {
		return fmt.Errorf("router: expected a Wintun device, got %T", dev)
	}
	luid := winipcfg.LUID(nt.LUID())

	// Assigning the mesh prefix (not a /32) makes Windows derive the
	// on-link route for the whole mesh, matching what `ip addr add
	// 100.64.x.y/10` does on Linux.
	if err := luid.SetIPAddresses([]netip.Prefix{prefix}); err != nil {
		return fmt.Errorf("router: set address %v on %s: %w", prefix, ifName, err)
	}

	iface, err := luid.IPInterface(winipcfg.AddressFamily(windowsAFInet))
	if err != nil {
		return fmt.Errorf("router: read interface %s: %w", ifName, err)
	}
	iface.NLMTU = uint32(mtu)
	if err := iface.Set(); err != nil {
		return fmt.Errorf("router: set mtu %d on %s: %w", mtu, ifName, err)
	}
	return nil
}

// AF_INET, spelled out to avoid importing x/sys/windows here.
const windowsAFInet = 2
