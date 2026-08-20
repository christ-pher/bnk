package magicsock

import (
	"errors"
	"net/netip"
	"strings"
	"testing"

	"github.com/christ-pher/bnk/internal/disco"
)

func TestEventLogRecordsDiscoTraffic(t *testing.T) {
	h := newPMHarness(t)
	h.setPeerAndProbe()

	// Answer the ping that went to cand1.
	for _, rp := range h.raw() {
		if rp.to == cand1 {
			txid := h.openAll([]rawPkt{rp})[0].(disco.Ping).TxID
			pong := disco.Seal(disco.Pong{TxID: txid, Observed: cand1}, h.pPriv, h.pPub, h.pub)
			h.pm.HandleDisco(cand1, pong)
		}
	}

	log := strings.Join(h.pm.Events(), "\n")
	for _, want := range []string{"cmm-tx", "ping-tx " + cand1.String(), "pong-rx " + cand1.String(), "promote " + cand1.String()} {
		if !strings.Contains(log, want) {
			t.Errorf("event log missing %q:\n%s", want, log)
		}
	}
}

func TestEventLogRecordsSendErrors(t *testing.T) {
	h := newPMHarness(t)
	h.mu.Lock()
	// Make every raw send fail from now on.
	h.pm.cfg.SendRaw = func(to netip.AddrPort, pkt []byte) error {
		return errors.New("sendto: network is unreachable")
	}
	h.mu.Unlock()
	h.setPeerAndProbe()

	log := strings.Join(h.pm.Events(), "\n")
	if !strings.Contains(log, "unreachable") {
		t.Errorf("send errors not recorded:\n%s", log)
	}
}
