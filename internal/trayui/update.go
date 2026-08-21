package trayui

import "fmt"

// UpdateState is what the tray currently knows about newer releases.
type UpdateState int

const (
	// UpdateUnknown is no answer: nothing has been checked yet, the
	// last check failed, or the last one found nothing and has since
	// gone stale. All three mean the same thing to a reader — the tray
	// cannot honestly say anything about newer releases — so they share
	// a label.
	UpdateUnknown UpdateState = iota
	// UpdateChecking is a check the user asked for, still running.
	UpdateChecking
	// UpdateFound is a newer release the tray can install.
	UpdateFound
	// UpdateCurrent is a check the user just asked for coming back
	// empty. It decays to UpdateUnknown, because it stops being true
	// the moment a release is published.
	UpdateCurrent
)

// UpdateLabel renders the update menu item.
//
// The resting label names the running version rather than claiming to
// be up to date. The claim was only ever true as of the last check, and
// the tray rechecks every few hours: long enough that it was usually
// wrong by the time anyone read it, which made the menu look broken
// until it was clicked. A version number is a fact and does not decay.
func UpdateLabel(state UpdateState, running, latest string) string {
	switch state {
	case UpdateChecking:
		return "Checking…"
	case UpdateFound:
		return "Update to " + latest
	case UpdateCurrent:
		return fmt.Sprintf("Up to date (%s)", running)
	}
	return fmt.Sprintf("Check for updates (%s)", running)
}
