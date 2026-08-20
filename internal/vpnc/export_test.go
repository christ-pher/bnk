package vpnc

import "net/netip"

// StateForTest exposes the persisted identity so tests can assert what
// survives a restart.
type StateForTest struct {
	IP     netip.Addr
	Prefix netip.Prefix
}

func LoadStateForTest(dir string) (StateForTest, bool, error) {
	st, ok, err := loadState(dir)
	return StateForTest{IP: st.IP, Prefix: st.Prefix}, ok, err
}
