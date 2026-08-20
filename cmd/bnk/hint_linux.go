//go:build linux

package main

import "fmt"

// adminHint explains the most common cause of a failed TUN creation.
func adminHint(err error) error {
	return fmt.Errorf("root required? %w", err)
}
