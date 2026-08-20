//go:build linux

package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/christ-pher/bnk/internal/vpnc"
)

// runDaemon runs the client until interrupted.
func runDaemon(cfg vpnc.Config) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	err := vpnc.Run(ctx, cfg)
	if ctx.Err() != nil {
		return nil // clean shutdown on signal
	}
	return err
}
