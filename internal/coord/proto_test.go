package coord

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/christ-pher/bnk/internal/netmap"
)

func testKey() netmap.Key {
	var k netmap.Key
	k[7] = 0x99
	return k
}

func TestFrameRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	payload := []byte(`{"t":"hello"}`)
	if err := WriteFrame(&buf, FrameControl, payload); err != nil {
		t.Fatal(err)
	}
	typ, got, err := ReadFrame(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if typ != FrameControl {
		t.Errorf("type = %v, want FrameControl", typ)
	}
	if !bytes.Equal(got, payload) {
		t.Errorf("payload = %q, want %q", got, payload)
	}
}

func TestFrameSequencePreservesBoundaries(t *testing.T) {
	var buf bytes.Buffer
	WriteFrame(&buf, FrameControl, []byte("first"))
	WriteFrame(&buf, FrameRelayData, []byte("second"))

	typ1, p1, err1 := ReadFrame(&buf)
	typ2, p2, err2 := ReadFrame(&buf)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	if typ1 != FrameControl || string(p1) != "first" {
		t.Errorf("frame 1 = %v %q", typ1, p1)
	}
	if typ2 != FrameRelayData || string(p2) != "second" {
		t.Errorf("frame 2 = %v %q", typ2, p2)
	}
}

func TestWriteFrameRejectsOversizePayload(t *testing.T) {
	err := WriteFrame(io.Discard, FrameControl, make([]byte, MaxFrameSize+1))
	if err == nil {
		t.Error("oversize write succeeded, want error")
	}
}

func TestReadFrameRejectsOversizeHeader(t *testing.T) {
	// Header claiming 2^24-1 bytes, above MaxFrameSize.
	hdr := []byte{byte(FrameControl), 0xFF, 0xFF, 0xFF}
	if _, _, err := ReadFrame(bytes.NewReader(hdr)); err == nil {
		t.Error("oversize read succeeded, want error")
	}
}

func TestReadFrameTruncatedStream(t *testing.T) {
	var buf bytes.Buffer
	WriteFrame(&buf, FrameControl, []byte("hello"))
	truncated := buf.Bytes()[:buf.Len()-2]
	if _, _, err := ReadFrame(bytes.NewReader(truncated)); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("err = %v, want io.ErrUnexpectedEOF", err)
	}
}

func TestControlEnvelopeRoundTrip(t *testing.T) {
	msg := Envelope{T: MsgHello, Hello: &Hello{NodeKey: testKey()}}
	raw, err := EncodeControl(msg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"hello"`) {
		t.Errorf("encoded control = %s, want tagged with \"hello\"", raw)
	}
	got, err := DecodeControl(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got.T != MsgHello || got.Hello == nil || got.Hello.NodeKey != msg.Hello.NodeKey {
		t.Errorf("decoded = %+v, want %+v", got, msg)
	}
}
