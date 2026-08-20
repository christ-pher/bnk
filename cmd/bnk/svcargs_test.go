package main

import (
	"slices"
	"testing"
)

func TestServiceArgsIncludesKeyOnFirstInstall(t *testing.T) {
	got := serviceArgs("https://vps:8443", "bnkkey:secret:fp", `C:\ProgramData\bnk`)
	want := []string{"run", "--server", "https://vps:8443", "--state-dir", `C:\ProgramData\bnk`, "--key", "bnkkey:secret:fp"}
	if !slices.Equal(got, want) {
		t.Errorf("serviceArgs = %v, want %v", got, want)
	}
}

// After enrollment the installer re-registers without the key so a spent
// key is never resubmitted.
func TestServiceArgsOmitsEmptyKey(t *testing.T) {
	got := serviceArgs("https://vps:8443", "", `C:\ProgramData\bnk`)
	if slices.Contains(got, "--key") {
		t.Errorf("serviceArgs = %v, want no --key when the key is empty", got)
	}
	want := []string{"run", "--server", "https://vps:8443", "--state-dir", `C:\ProgramData\bnk`}
	if !slices.Equal(got, want) {
		t.Errorf("serviceArgs = %v, want %v", got, want)
	}
}
