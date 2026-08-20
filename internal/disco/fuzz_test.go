package disco

import "testing"

// FuzzOpen asserts the parser never panics on hostile input.
func FuzzOpen(f *testing.F) {
	aPriv, aPub, _, bPub := [32]byte{1}, [32]byte{2}, [32]byte{3}, [32]byte{4}
	f.Add(Seal(Ping{TxID: [12]byte{7}}, aPriv, aPub, bPub))
	f.Add([]byte(Magic))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, pkt []byte) {
		Open(pkt, [32]byte{9})
	})
}
