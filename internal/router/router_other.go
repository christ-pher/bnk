//go:build !linux && !windows

package router

import (
	"fmt"
	"net/netip"
	"runtime"

	"golang.zx2c4.com/wireguard/tun"
)

func Up(dev tun.Device, ifName string, prefix netip.Prefix, mtu int) error {
	return fmt.Errorf("router: %s is not supported yet (Linux and Windows only)", runtime.GOOS)
}
