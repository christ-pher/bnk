//go:build !linux

package router

import (
	"fmt"
	"net/netip"
	"runtime"
)

func Up(ifName string, prefix netip.Prefix, mtu int) error {
	return fmt.Errorf("router: %s is not supported yet (Linux only for now)", runtime.GOOS)
}
