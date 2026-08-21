package vpnc

import (
	"context"
	"crypto/tls"
	"fmt"
	"strings"
	"time"

	"github.com/christ-pher/bnk/internal/coord/client"
	"github.com/christ-pher/bnk/internal/netmap"
	"github.com/christ-pher/bnk/internal/pin"
)

// Leave asks the control server to forget this node, using the stored
// identity to authenticate. It is best-effort by design: uninstalling
// must succeed even when the server is unreachable, in which case the
// operator clears the entry with `bnk-server node rm`.
func Leave(ctx context.Context, stateDir string) error {
	st, ok, err := loadState(stateDir)
	if err != nil {
		return err
	}
	if !ok || st.NodeID == 0 {
		return fmt.Errorf("this machine is not enrolled; nothing to leave")
	}

	var tlsConf *tls.Config
	if strings.HasPrefix(st.ServerURL, "https://") {
		if st.Fingerprint == "" {
			return fmt.Errorf("no pinned fingerprint in state")
		}
		tlsConf = pin.ClientTLSConfig(st.Fingerprint)
	}

	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	sess, err := client.Dial(ctx, st.ServerURL, tlsConf, st.PrivateKey, client.Handlers{
		OnNetmap: func(netmap.Netmap) {},
	})
	if err != nil {
		return fmt.Errorf("contacting %s: %w", st.ServerURL, err)
	}
	defer sess.Close()
	if err := sess.Leave(); err != nil {
		return err
	}
	// The server drops the session once it has removed the node, so the
	// close is the acknowledgement. Waiting for it matters because the
	// uninstaller tears the machine down the moment this returns.
	select {
	case <-sess.Done():
		return nil
	case <-ctx.Done():
		return fmt.Errorf("left, but the server did not confirm in time")
	}
}
