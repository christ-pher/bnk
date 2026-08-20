// Package ipam allocates tunnel addresses (/32s) from a prefix.
package ipam

import (
	"fmt"
	"net/netip"
)

// Next returns the lowest free host address in prefix. The network address
// and the prefix's last (broadcast) address are never allocated.
func Next(prefix netip.Prefix, used map[netip.Addr]bool) (netip.Addr, error) {
	prefix = prefix.Masked()
	last := lastAddr(prefix)
	for a := prefix.Addr().Next(); a.Less(last); a = a.Next() {
		if !used[a] {
			return a, nil
		}
	}
	return netip.Addr{}, fmt.Errorf("ipam: prefix %v exhausted", prefix)
}

func lastAddr(p netip.Prefix) netip.Addr {
	bytes := p.Addr().As4()
	hostBits := 32 - p.Bits()
	for i := 0; i < hostBits; i++ {
		bytes[3-i/8] |= 1 << (i % 8)
	}
	return netip.AddrFrom4(bytes)
}
