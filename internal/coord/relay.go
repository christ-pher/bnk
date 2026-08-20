package coord

import (
	"encoding/binary"
	"errors"

	"vpnmesh/internal/netmap"
)

// Relay frame payload: [4B big-endian NodeID][WireGuard packet]. The ID is
// the destination client→server and the source server→client, so a sender
// can never spoof its identity.

func EncodeRelay(id netmap.NodeID, pkt []byte) []byte {
	out := make([]byte, 4+len(pkt))
	binary.BigEndian.PutUint32(out, uint32(id))
	copy(out[4:], pkt)
	return out
}

func DecodeRelay(payload []byte) (netmap.NodeID, []byte, error) {
	if len(payload) < 4 {
		return 0, nil, errors.New("coord: relay payload shorter than its header")
	}
	return netmap.NodeID(binary.BigEndian.Uint32(payload)), payload[4:], nil
}
