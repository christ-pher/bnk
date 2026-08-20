package vpnc

import "testing"

func TestControlSDDLWithoutOperatorIsAdminOnly(t *testing.T) {
	got, err := controlSDDL("")
	if err != nil {
		t.Fatal(err)
	}
	want := "D:P(A;;GA;;;SY)(A;;GA;;;BA)"
	if got != want {
		t.Errorf("controlSDDL(\"\") = %q, want %q", got, want)
	}
}

func TestControlSDDLGrantsTheOperator(t *testing.T) {
	sid := "S-1-5-21-1004336348-1177238915-682003330-1001"
	got, err := controlSDDL(sid)
	if err != nil {
		t.Fatal(err)
	}
	want := "D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;" + sid + ")"
	if got != want {
		t.Errorf("controlSDDL(%q) = %q, want %q", sid, got, want)
	}
}

// The operator string ends up inside an access control list, so anything
// that is not plainly a SID must be refused rather than spliced in.
func TestControlSDDLRejectsAnythingThatIsNotASID(t *testing.T) {
	for _, bad := range []string{
		"WD",                        // a well-known alias, not a SID
		"S-1-5-21-1001)(A;;GA;;;WD", // ACE injection
		"S-1-",
		"S-1-5-21-abc",
		"nonsense",
		"s-1-5-21-1001", // lowercase
		"S-1-5-21-1001 ",
	} {
		if _, err := controlSDDL(bad); err == nil {
			t.Errorf("controlSDDL(%q) was accepted, want an error", bad)
		}
	}
}
