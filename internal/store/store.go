// Package store persists control-server state as a single JSON file,
// written atomically (temp file + rename).
package store

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/netip"
	"os"
	"path/filepath"
	"time"

	"vpnmesh/internal/acl"
	"vpnmesh/internal/netmap"
)

type State struct {
	Prefix     netip.Prefix `json:"prefix"`
	Nodes      []Node       `json:"nodes"`
	EnrollKeys []EnrollKey  `json:"enroll_keys"`
	Policy     *acl.Policy  `json:"policy,omitempty"`
}

type Node struct {
	ID        netmap.NodeID `json:"id"`
	Name      string        `json:"name"`
	OS        string        `json:"os,omitempty"`
	NodeKey   netmap.Key    `json:"node_key"`
	DiscoKey  netmap.Key    `json:"disco_key"`
	IP        netip.Addr    `json:"ip"`
	Tags      []string      `json:"tags,omitempty"`
	CreatedAt time.Time     `json:"created_at"`
}

type EnrollKey struct {
	Secret    string    `json:"secret"`
	Revoked   bool      `json:"revoked"`
	CreatedAt time.Time `json:"created_at"`
}

type FileStore struct {
	path string
}

func NewFileStore(path string) *FileStore {
	return &FileStore{path: path}
}

func (s *FileStore) Load() (State, error) {
	raw, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return State{}, nil
	}
	if err != nil {
		return State{}, err
	}
	var st State
	if err := json.Unmarshal(raw, &st); err != nil {
		return State{}, fmt.Errorf("state file %s: %w", s.path, err)
	}
	return st, nil
}

func (s *FileStore) Save(st State) error {
	raw, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".state-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), s.path)
}
