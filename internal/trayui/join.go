package trayui

import (
	"fmt"
	"regexp"
	"strings"
)

// Join is what a user pasted into the sign-in box, reduced to the two
// things enrolment needs.
type Join struct {
	Server string // may be empty: the daemon may already know its server
	Key    string
}

var (
	// bnkkey:<secret>:<fingerprint> — both halves required, so a
	// truncated paste is rejected rather than failing later at enrol.
	keyPattern = regexp.MustCompile(`bnkkey:[^\s:"']+:[^\s:"']+`)

	// A control server URL, with the port it is always deployed with.
	// Requiring a port is what separates it from the installer URL that
	// appears in the same pasted line.
	serverPattern = regexp.MustCompile(`https://[A-Za-z0-9._-]+:\d+`)
)

// ParseJoin pulls an enrolment key, and a server if one is present, out
// of pasted text. It accepts the whole command `bnk-server key new`
// prints for either platform, or just the key on its own, because both
// are things a person plausibly copies.
func ParseJoin(pasted string) (Join, error) {
	s := strings.TrimSpace(pasted)
	if s == "" {
		return Join{}, fmt.Errorf("nothing pasted")
	}
	key := keyPattern.FindString(s)
	if key == "" {
		return Join{}, fmt.Errorf("no enrolment key found — paste the whole join command, or the bnkkey:... line, from `bnk-server key new`")
	}
	return Join{Server: serverPattern.FindString(s), Key: key}, nil
}
