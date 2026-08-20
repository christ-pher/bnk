package ipam

import (
	"encoding/binary"
	"fmt"
	"net/netip"
)

// Reassign maps every address in addrs from one prefix into another,
// keeping each address's host offset when it still fits so a node at .3
// stays at .3, and packing the rest into the lowest free addresses. The
// network and broadcast addresses are never assigned.
func Reassign(old, current netip.Prefix, addrs []netip.Addr) (map[netip.Addr]netip.Addr, error) {
	old, current = old.Masked(), current.Masked()
	if !current.Addr().Is4() {
		return nil, fmt.Errorf("ipam: %v is not IPv4", current)
	}
	if capacity(current) < uint64(len(addrs)) {
		return nil, fmt.Errorf("ipam: %v holds %d addresses but %d nodes need one", current, capacity(current), len(addrs))
	}

	out := make(map[netip.Addr]netip.Addr, len(addrs))
	taken := make(map[netip.Addr]bool, len(addrs))
	base := toU32(current.Addr())
	last := toU32(lastAddr(current))

	// First pass: everyone whose host offset still lands inside the new
	// network keeps it.
	var packed []netip.Addr
	for _, a := range addrs {
		if !a.Is4() {
			return nil, fmt.Errorf("ipam: %v is not IPv4", a)
		}
		want := base + (toU32(a) - toU32(old.Addr()))
		candidate := fromU32(want)
		if want > base && want < last && !taken[candidate] {
			out[a] = candidate
			taken[candidate] = true
			continue
		}
		packed = append(packed, a)
	}

	// Second pass: everyone else takes the lowest address still free.
	for _, a := range packed {
		next, err := Next(current, taken)
		if err != nil {
			return nil, err
		}
		out[a] = next
		taken[next] = true
	}
	return out, nil
}

// capacity is how many host addresses a prefix can assign, excluding the
// network and broadcast addresses.
func capacity(p netip.Prefix) uint64 {
	hostBits := 32 - p.Bits()
	if hostBits < 2 {
		return 0
	}
	return uint64(1)<<uint(hostBits) - 2
}

func toU32(a netip.Addr) uint32 {
	b := a.As4()
	return binary.BigEndian.Uint32(b[:])
}

func fromU32(v uint32) netip.Addr {
	var b [4]byte
	binary.BigEndian.PutUint32(b[:], v)
	return netip.AddrFrom4(b)
}
