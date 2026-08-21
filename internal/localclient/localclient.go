// Package localclient talks to a running bnk daemon over its local API.
// Both the CLI and the tray app use it, so the transport (unix socket on
// Linux, named pipes on Windows) is defined in exactly one place.
package localclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"

	"github.com/christ-pher/bnk/internal/vpnc"
)

// New returns a client bound to one local API endpoint.
func New(endpoint string) *http.Client {
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return dial(ctx, endpoint)
		},
	}}
}

// Get fetches a diagnostics endpoint and decodes it into out.
func Get(endpoint, path string, out any) error {
	resp, err := New(endpoint).Get("http://bnk" + path)
	if err != nil {
		return fmt.Errorf("is the bnk daemon running? %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errorFrom(resp)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Status reports the daemon's view of the mesh.
func Status(endpoint string) (vpnc.Status, error) {
	var st vpnc.Status
	err := Get(endpoint, "/status", &st)
	return st, err
}

// Post sends a control verb. Control travels on its own endpoint: a
// separate, restricted pipe on Windows; the same socket on Linux.
func Post(endpoint, path string) error {
	resp, err := New(ControlEndpoint(endpoint)).Post("http://bnk"+path, "application/json", nil)
	if err != nil {
		return fmt.Errorf("%s (%w)", DaemonDownHint, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errorFrom(resp)
	}
	return nil
}

// Join enrols this machine into a mesh. server may be empty when the
// daemon already knows which server it belongs to.
func Join(endpoint, server, key string) error {
	body, err := json.Marshal(map[string]string{"server": server, "key": key})
	if err != nil {
		return err
	}
	resp, err := New(ControlEndpoint(endpoint)).Post("http://bnk/join", "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("%s (%w)", DaemonDownHint, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return errorFrom(resp)
	}
	return nil
}

func Up(endpoint string) error   { return Post(endpoint, "/up") }
func Down(endpoint string) error { return Post(endpoint, "/down") }

func errorFrom(resp *http.Response) error {
	msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	return fmt.Errorf("%s", bytes.TrimSpace(msg))
}
