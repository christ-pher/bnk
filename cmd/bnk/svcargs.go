package main

import "strings"

// serviceArgs builds the argument list the Windows service manager
// stores for the daemon. It lives in a portable file so its behavior —
// especially dropping a spent enrollment key — is testable anywhere.
//
// The key is single-use: the installer registers the service with it,
// then re-registers without it once the node has enrolled, mirroring how
// the Linux installer blanks BNK_KEY in the env file.
func serviceArgs(server, key, stateDir, operatorSID string) []string {
	args := []string{"run", "--state-dir", stateDir}
	if server != "" {
		args = append(args, "--server", server)
	}
	if operatorSID != "" {
		args = append(args, "--operator", operatorSID)
	}
	if key != "" {
		args = append(args, "--key", key)
	}
	return args
}

// operatorFromCommandLine recovers the --operator value from a
// registered service command line, so uninstall can undo what install
// did without being told again.
func operatorFromCommandLine(cmd string) string {
	fields := strings.Fields(cmd)
	for i, f := range fields {
		if f == "--operator" && i+1 < len(fields) {
			return strings.Trim(fields[i+1], `"`)
		}
	}
	return ""
}
