package filter

import (
	"os"
	"testing"

	"golang.zx2c4.com/wireguard/tun"

	"vpnmesh/internal/acl"
)

// fakeTUN records writes and serves queued reads.
type fakeTUN struct {
	written [][]byte
	pending [][]byte
	events  chan tun.Event
}

func (f *fakeTUN) File() *os.File { return nil }
func (f *fakeTUN) Read(bufs [][]byte, sizes []int, offset int) (int, error) {
	n := 0
	for n < len(bufs) && len(f.pending) > 0 {
		pkt := f.pending[0]
		f.pending = f.pending[1:]
		copy(bufs[n][offset:], pkt)
		sizes[n] = len(pkt)
		n++
	}
	return n, nil
}
func (f *fakeTUN) Write(bufs [][]byte, offset int) (int, error) {
	for _, b := range bufs {
		f.written = append(f.written, append([]byte(nil), b[offset:]...))
	}
	return len(bufs), nil
}
func (f *fakeTUN) MTU() (int, error)        { return 1280, nil }
func (f *fakeTUN) Name() (string, error)    { return "fake0", nil }
func (f *fakeTUN) Events() <-chan tun.Event { return f.events }
func (f *fakeTUN) BatchSize() int           { return 4 }
func (f *fakeTUN) Close() error             { return nil }

func TestWrapTUNDropsDeniedInboundWrites(t *testing.T) {
	inner := &fakeTUN{events: make(chan tun.Event)}
	f := sshOnlyFilter()
	wrapped := WrapTUN(inner, f)

	allowed := tcp(peer, self, 5555, 22, true)
	denied := tcp(peer, self, 5555, 23, true)
	offset := 16
	pad := func(p []byte) []byte { return append(make([]byte, offset), p...) }

	n, err := wrapped.Write([][]byte{pad(allowed), pad(denied)}, offset)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("Write returned %d, want 2 (denied packets are silently dropped)", n)
	}
	if len(inner.written) != 1 {
		t.Fatalf("inner TUN got %d packets, want only the allowed one", len(inner.written))
	}
	if string(inner.written[0]) != string(allowed) {
		t.Error("the surviving packet is not the allowed one")
	}
}

func TestWrapTUNReadRecordsOutboundFlows(t *testing.T) {
	inner := &fakeTUN{events: make(chan tun.Event)}
	f := New()
	f.SetRules([]acl.CompiledRule{}) // deny-all policy: only flows admit traffic
	wrapped := WrapTUN(inner, f)

	// The local host sends a UDP packet out through the TUN...
	out := udp(self, peer, 40000, 9999)
	inner.pending = append(inner.pending, out)
	bufs := [][]byte{make([]byte, 2048)}
	sizes := make([]int, 1)
	if _, err := wrapped.Read(bufs, sizes, 16); err != nil {
		t.Fatal(err)
	}

	// ...so the reply must pass, and unrelated inbound must not.
	reply := udp(peer, self, 9999, 40000)
	unrelated := udp(peer, self, 9999, 40001)
	offset := 16
	pad := func(p []byte) []byte { return append(make([]byte, offset), p...) }
	if _, err := wrapped.Write([][]byte{pad(reply), pad(unrelated)}, offset); err != nil {
		t.Fatal(err)
	}
	if len(inner.written) != 1 || string(inner.written[0]) != string(reply) {
		t.Fatalf("inner got %d packets, want just the reply", len(inner.written))
	}
}
