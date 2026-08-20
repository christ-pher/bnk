package store

import (
	"net/netip"
	"os"
	"path/filepath"
	"testing"

	"vpnmesh/internal/netmap"
)

func TestLoadMissingFileReturnsEmptyState(t *testing.T) {
	fs := NewFileStore(filepath.Join(t.TempDir(), "state.json"))
	st, err := fs.Load()
	if err != nil {
		t.Fatalf("Load on first run: %v", err)
	}
	if len(st.Nodes) != 0 || len(st.EnrollKeys) != 0 {
		t.Errorf("empty state has nodes=%v keys=%v", st.Nodes, st.EnrollKeys)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")
	fs := NewFileStore(path)

	var key netmap.Key
	key[5] = 0x42
	want := State{
		Prefix: netip.MustParsePrefix("100.64.0.0/10"),
		Nodes: []Node{{
			ID:      1,
			Name:    "laptop",
			NodeKey: key,
			IP:      netip.MustParseAddr("100.64.0.1"),
			Tags:    []string{"trusted"},
		}},
		EnrollKeys: []EnrollKey{{Secret: "s3cret", Revoked: false}},
	}
	if err := fs.Save(want); err != nil {
		t.Fatal(err)
	}

	got, err := NewFileStore(path).Load()
	if err != nil {
		t.Fatal(err)
	}
	if got.Prefix != want.Prefix {
		t.Errorf("Prefix = %v, want %v", got.Prefix, want.Prefix)
	}
	if len(got.Nodes) != 1 || got.Nodes[0].Name != "laptop" || got.Nodes[0].NodeKey != key || got.Nodes[0].IP != want.Nodes[0].IP {
		t.Errorf("Nodes = %+v, want %+v", got.Nodes, want.Nodes)
	}
	if len(got.EnrollKeys) != 1 || got.EnrollKeys[0].Secret != "s3cret" {
		t.Errorf("EnrollKeys = %+v, want %+v", got.EnrollKeys, want.EnrollKeys)
	}
}

func TestSaveLeavesNoTempFiles(t *testing.T) {
	dir := t.TempDir()
	fs := NewFileStore(filepath.Join(dir, "state.json"))
	if err := fs.Save(State{}); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "state.json" {
		names := []string{}
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("dir contains %v, want only state.json", names)
	}
}
