package main

import (
	"slices"
	"testing"
)

func TestServiceArgsIncludesKeyOnFirstInstall(t *testing.T) {
	got := serviceArgs("https://vps:8443", "bnkkey:secret:fp", `C:\ProgramData\bnk`, "")
	want := []string{"run", "--server", "https://vps:8443", "--state-dir", `C:\ProgramData\bnk`, "--key", "bnkkey:secret:fp"}
	if !slices.Equal(got, want) {
		t.Errorf("serviceArgs = %v, want %v", got, want)
	}
}

// After enrollment the installer re-registers without the key so a spent
// key is never resubmitted.
func TestServiceArgsOmitsEmptyKey(t *testing.T) {
	got := serviceArgs("https://vps:8443", "", `C:\ProgramData\bnk`, "")
	if slices.Contains(got, "--key") {
		t.Errorf("serviceArgs = %v, want no --key when the key is empty", got)
	}
	want := []string{"run", "--server", "https://vps:8443", "--state-dir", `C:\ProgramData\bnk`}
	if !slices.Equal(got, want) {
		t.Errorf("serviceArgs = %v, want %v", got, want)
	}
}

// The operator SID has to survive the key being scrubbed after
// enrollment, or the tray loses its permission on the second install.
func TestServiceArgsKeepsOperatorWithoutKey(t *testing.T) {
	sid := "S-1-5-21-1-2-3-1001"
	got := serviceArgs("https://vps:8443", "", `C:\ProgramData\bnk`, sid)
	want := []string{"run", "--server", "https://vps:8443", "--state-dir", `C:\ProgramData\bnk`, "--operator", sid}
	if !slices.Equal(got, want) {
		t.Errorf("serviceArgs = %v, want %v", got, want)
	}
}

func TestOperatorFromCommandLine(t *testing.T) {
	sid := "S-1-5-21-1-2-3-1001"
	cases := map[string]string{
		`"C:\Program Files\bnk\bnk.exe" run --server https://v:8443 --state-dir "C:\ProgramData\bnk" --operator ` + sid: sid,
		`"C:\Program Files\bnk\bnk.exe" run --server https://v:8443`:                                                    "",
		`"C:\bnk.exe" run --operator "` + sid + `"`:                                                                     sid,
		`"C:\bnk.exe" run --operator`:                                                                                   "",
	}
	for cmd, want := range cases {
		if got := operatorFromCommandLine(cmd); got != want {
			t.Errorf("operatorFromCommandLine(%q) = %q, want %q", cmd, got, want)
		}
	}
}
