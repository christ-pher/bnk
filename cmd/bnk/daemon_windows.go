//go:build windows

package main

import (
	"context"
	"os"
	"os/signal"

	"golang.org/x/sys/windows/svc"

	"github.com/christ-pher/bnk/internal/vpnc"
)

const serviceName = "bnk"

// runDaemon runs under the Service Control Manager when started as a
// service, and as an ordinary foreground process otherwise (which is how
// it behaves when run by hand for debugging).
func runDaemon(cfg vpnc.Config) error {
	isService, err := svc.IsWindowsService()
	if err != nil {
		return err
	}
	if !isService {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		err := vpnc.Run(ctx, cfg)
		if ctx.Err() != nil {
			return nil
		}
		return err
	}
	return svc.Run(serviceName, &windowsService{cfg: cfg})
}

type windowsService struct{ cfg vpnc.Config }

// Execute satisfies svc.Handler: it reports state transitions to the SCM
// and cancels the daemon on Stop or system shutdown.
func (s *windowsService) Execute(args []string, req <-chan svc.ChangeRequest, status chan<- svc.Status) (bool, uint32) {
	const accepted = svc.AcceptStop | svc.AcceptShutdown
	status <- svc.Status{State: svc.StartPending}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- vpnc.Run(ctx, s.cfg) }()

	status <- svc.Status{State: svc.Running, Accepts: accepted}
	for {
		select {
		case c := <-req:
			switch c.Cmd {
			case svc.Interrogate:
				status <- c.CurrentStatus
			case svc.Stop, svc.Shutdown:
				status <- svc.Status{State: svc.StopPending}
				cancel()
				<-done
				status <- svc.Status{State: svc.Stopped}
				return false, 0
			}
		case err := <-done:
			// The daemon exited on its own: report failure so the SCM's
			// restart policy can act.
			status <- svc.Status{State: svc.Stopped}
			if err != nil && ctx.Err() == nil {
				return false, 1
			}
			return false, 0
		}
	}
}
