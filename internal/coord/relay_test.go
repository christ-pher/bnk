package coord

import (
	"bytes"
	"testing"
)

func TestRelayHeaderRoundTrip(t *testing.T) {
	pkt := []byte{0x01, 0x02, 0x03}
	framed := EncodeRelay(0xDEADBEEF, pkt)
	id, got, err := DecodeRelay(framed)
	if err != nil {
		t.Fatal(err)
	}
	if id != 0xDEADBEEF {
		t.Errorf("id = %#x, want 0xDEADBEEF", id)
	}
	if !bytes.Equal(got, pkt) {
		t.Errorf("packet = %v, want %v", got, pkt)
	}
}

func TestDecodeRelayRejectsShortPayload(t *testing.T) {
	if _, _, err := DecodeRelay([]byte{1, 2}); err == nil {
		t.Error("short payload decoded, want error")
	}
}
