package main

// serviceArgs builds the argument list the Windows service manager
// stores for the daemon. It lives in a portable file so its behavior —
// especially dropping a spent enrollment key — is testable anywhere.
//
// The key is single-use: the installer registers the service with it,
// then re-registers without it once the node has enrolled, mirroring how
// the Linux installer blanks BNK_KEY in the env file.
func serviceArgs(server, key, stateDir string) []string {
	args := []string{"run", "--server", server, "--state-dir", stateDir}
	if key != "" {
		args = append(args, "--key", key)
	}
	return args
}
