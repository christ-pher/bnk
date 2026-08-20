//go:build windows

package vpnc

import (
	"io"
	"net/http"

	"github.com/Microsoft/go-winio"
)

// Windows has no SO_PEERCRED, so the split that Linux enforces with a
// uid check is enforced here by the pipes' ACLs instead: diagnostics
// live on a pipe any local user may open, control verbs on one only
// Administrators and SYSTEM may open. A non-elevated caller cannot open
// the control pipe at all — UAC leaves the Administrators SID deny-only
// in an unelevated token.
// ControlPipe returns the control pipe name paired with a diagnostics
// pipe name, so the daemon and CLI derive it the same way.
func ControlPipe(diagnostics string) string {
	return diagnostics + "-ctl"
}

// serveLocalAPI exposes daemon state to the CLI over two named pipes.
// sock is the diagnostics pipe name (e.g. \\.\pipe\bnk); the control
// pipe is derived from it. The caller owns the returned closer.
func serveLocalAPI(sock string, c *controller) (io.Closer, error) {
	diagLn, err := winio.ListenPipe(sock, &winio.PipeConfig{SecurityDescriptor: sddlDiagnostics})
	if err != nil {
		return nil, err
	}
	ctlSDDL, err := controlSDDL(c.cfg.OperatorSID)
	if err != nil {
		diagLn.Close()
		return nil, err
	}
	ctlLn, err := winio.ListenPipe(ControlPipe(sock), &winio.PipeConfig{SecurityDescriptor: ctlSDDL})
	if err != nil {
		diagLn.Close()
		return nil, err
	}

	diagMux := http.NewServeMux()
	registerDiagnostics(diagMux, c)
	ctlMux := http.NewServeMux()
	// No gate: opening the control pipe already proved the caller is an
	// administrator.
	registerControl(ctlMux, c, nil)

	go http.Serve(diagLn, diagMux)
	go http.Serve(ctlLn, ctlMux)
	return multiCloser{diagLn, ctlLn}, nil
}

type multiCloser []io.Closer

func (m multiCloser) Close() error {
	var firstErr error
	for _, c := range m {
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
