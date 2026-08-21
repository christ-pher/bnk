package trayui_test

import (
	"strings"
	"testing"

	"github.com/christ-pher/bnk/internal/trayui"
)

// The resting label must state the running version rather than claim
// freshness. A claim is only true as of the last check, and the tray
// rechecks every six hours — long enough that "Up to date" was usually
// a lie by the time anyone read it.
func TestUpdateLabelAtRestNamesTheRunningVersion(t *testing.T) {
	got := trayui.UpdateLabel(trayui.UpdateUnknown, "v0.10.1", "")
	if !strings.Contains(got, "v0.10.1") {
		t.Errorf("label %q should name the running version", got)
	}
	if strings.Contains(strings.ToLower(got), "up to date") {
		t.Errorf("label %q claims freshness no check has established", got)
	}
	if !strings.Contains(strings.ToLower(got), "check for update") {
		t.Errorf("label %q should offer to check", got)
	}
}

// A failed check must fall back to the resting label, not to silence
// and not to a claim: the version is still known even when GitHub is not
// reachable.
func TestUpdateLabelAfterAFailedCheckStillNamesTheVersion(t *testing.T) {
	rest := trayui.UpdateLabel(trayui.UpdateUnknown, "v0.10.1", "")
	if got := trayui.UpdateLabel(trayui.UpdateUnknown, "v0.10.1", "v0.10.2"); got != rest {
		t.Errorf("label = %q, want the resting label %q", got, rest)
	}
}

func TestUpdateLabelOffersTheNewerRelease(t *testing.T) {
	got := trayui.UpdateLabel(trayui.UpdateFound, "v0.10.1", "v0.10.2")
	if !strings.Contains(got, "v0.10.2") {
		t.Errorf("label %q should name the release on offer", got)
	}
	if strings.Contains(got, "v0.10.1") {
		t.Errorf("label %q names the running version; the action is what matters", got)
	}
}

// "Up to date" is allowed only just after a check the user asked for,
// where it answers a question they just posed and is momentarily true.
func TestUpdateLabelConfirmsAFreshCheck(t *testing.T) {
	got := trayui.UpdateLabel(trayui.UpdateCurrent, "v0.10.1", "v0.10.1")
	if !strings.Contains(strings.ToLower(got), "up to date") {
		t.Errorf("label %q should confirm the check found nothing", got)
	}
	if !strings.Contains(got, "v0.10.1") {
		t.Errorf("label %q should name the version it is current with", got)
	}
}

func TestUpdateLabelSaysWhenItIsWorking(t *testing.T) {
	got := trayui.UpdateLabel(trayui.UpdateChecking, "v0.10.1", "")
	if !strings.Contains(got, "…") {
		t.Errorf("label %q should read as in-progress", got)
	}
}

// Local builds report "dev". Naming it is still the useful answer —
// it tells you which binary you are running.
func TestUpdateLabelHandlesLocalBuilds(t *testing.T) {
	if got := trayui.UpdateLabel(trayui.UpdateUnknown, "dev", ""); !strings.Contains(got, "dev") {
		t.Errorf("label %q should name the running build", got)
	}
}
