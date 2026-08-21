package trayicon

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// wantSizes is what packaging/icons/gen.py emits. Windows picks the
// frame closest to the current DPI, so a missing size is not a crash —
// it is a blurry icon nobody notices until it ships.
var wantSizes = []int{16, 20, 24, 32, 48, 64}

func TestIconsHaveEverySize(t *testing.T) {
	for name, ico := range map[string][]byte{
		"connected":    Connected,
		"disconnected": Disconnected,
		"attention":    Attention,
	} {
		got := icoSizes(t, ico)
		if len(got) != len(wantSizes) {
			t.Fatalf("%s: got %d frames %v, want %v", name, len(got), got, wantSizes)
		}
		for i, w := range wantSizes {
			if got[i] != w {
				t.Errorf("%s: frame %d is %dpx, want %dpx", name, i, got[i], w)
			}
		}
	}
}

// The three states are one generator apart, and the only thing that
// distinguishes them is a dot colour. Identical bytes would mean the
// tray silently shows the same icon for every state.
func TestStatesDiffer(t *testing.T) {
	for _, p := range []struct {
		a, b string
		x, y []byte
	}{
		{"connected", "disconnected", Connected, Disconnected},
		{"connected", "attention", Connected, Attention},
		{"disconnected", "attention", Disconnected, Attention},
	} {
		if bytes.Equal(p.x, p.y) {
			t.Errorf("%s and %s are the same image", p.a, p.b)
		}
	}
}

// icoSizes reads the ICONDIR at the head of an .ico and returns the
// width of every frame, ascending. It also checks each frame's bytes
// are actually present, which a truncated commit would not be.
func icoSizes(t *testing.T, b []byte) []int {
	t.Helper()
	if len(b) < 6 {
		t.Fatalf("too short to be an .ico: %d bytes", len(b))
	}
	if typ := binary.LittleEndian.Uint16(b[2:4]); typ != 1 {
		t.Fatalf("image type %d, want 1 (icon)", typ)
	}
	n := int(binary.LittleEndian.Uint16(b[4:6]))
	var sizes []int
	for i := 0; i < n; i++ {
		e := 6 + i*16
		if len(b) < e+16 {
			t.Fatalf("directory entry %d is truncated", i)
		}
		w := int(b[e])
		if w == 0 {
			w = 256 // the format stores 256 as zero
		}
		size := binary.LittleEndian.Uint32(b[e+8 : e+12])
		off := binary.LittleEndian.Uint32(b[e+12 : e+16])
		if size == 0 {
			t.Errorf("frame %d (%dpx) is empty", i, w)
		}
		if int(off)+int(size) > len(b) {
			t.Errorf("frame %d (%dpx) runs past the end of the file", i, w)
		}
		sizes = append(sizes, w)
	}
	for i := 1; i < len(sizes); i++ {
		if sizes[i] < sizes[i-1] {
			t.Errorf("frames are not ascending: %v", sizes)
			break
		}
	}
	return sizes
}
