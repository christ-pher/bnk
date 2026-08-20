package selfupdate

// UpdateAvailable reports whether the latest release differs from the
// running version, and what that release is.
//
// The comparison is deliberately "different", not "greater": versions
// are release tags, a locally built binary reports "dev", and offering
// the published release in that case is the useful answer.
func UpdateAvailable(baseURL, current string) (latest string, available bool, err error) {
	latest, err = LatestTag(baseURL)
	if err != nil {
		return "", false, err
	}
	return latest, latest != current, nil
}
