package vpnc

import (
	"fmt"
	"regexp"
)

// Security descriptors for the two local API pipes. Diagnostics are
// readable by any local user; control is not.
const (
	// SYSTEM and Administrators get full access; Everyone (WD) may read
	// and write, which is all a request/response client needs.
	sddlDiagnostics = "D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GRGW;;;WD)"
	sddlControlBase = "D:P(A;;GA;;;SY)(A;;GA;;;BA)"
)

// sidPattern matches a string SID: S-1-<authority>-<subauthority>... and
// nothing else. Operator SIDs are pasted straight into an ACL, so this
// is the boundary that keeps a crafted string from adding its own ACEs.
var sidPattern = regexp.MustCompile(`^S-1-\d+(-\d+)+$`)

// controlSDDL returns the security descriptor for the control pipe. An
// empty operator yields the administrators-only descriptor; otherwise
// the operator's SID is granted the same access, which is what lets a
// tray app toggle the tunnel without elevating.
func controlSDDL(operatorSID string) (string, error) {
	if operatorSID == "" {
		return sddlControlBase, nil
	}
	if !sidPattern.MatchString(operatorSID) {
		return "", fmt.Errorf("operator %q is not a SID (expected S-1-5-21-...)", operatorSID)
	}
	return sddlControlBase + "(A;;GA;;;" + operatorSID + ")", nil
}
