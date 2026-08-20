// Package dataplane assembles the client's packet path: TUN device,
// (later) ACL filter, WireGuard device, and magicsock Bind.
package dataplane

import (
	"encoding/hex"
	"fmt"
	"strings"
	"sync"

	"golang.zx2c4.com/wireguard/device"
	"golang.zx2c4.com/wireguard/tun"

	"vpnmesh/internal/magicsock"
	"vpnmesh/internal/netmap"
)

type Engine struct {
	mu   sync.Mutex
	bind *magicsock.Bind
	dev  *device.Device
}

// New brings up a WireGuard device on tunDev with a fresh magicsock Bind.
func New(tunDev tun.Device, privateKey [32]byte) (*Engine, error) {
	bind := magicsock.NewBind()
	dev := device.NewDevice(tunDev, bind, device.NewLogger(device.LogLevelError, "wg: "))
	if err := dev.IpcSet(fmt.Sprintf("private_key=%s\n", hex.EncodeToString(privateKey[:]))); err != nil {
		dev.Close()
		return nil, err
	}
	if err := dev.Up(); err != nil {
		dev.Close()
		return nil, err
	}
	return &Engine{bind: bind, dev: dev}, nil
}

func (e *Engine) LocalPort() uint16 {
	return e.bind.LocalPort()
}

// ApplyNetmap reconfigures the device and path table to match nm. Peers
// absent from nm are removed (replace_peers); each peer's identity is its
// node key, and its freshest known endpoint feeds the Bind's path table.
func (e *Engine) ApplyNetmap(nm netmap.Netmap) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	var cfg strings.Builder
	cfg.WriteString("replace_peers=true\n")
	for _, p := range nm.Peers {
		fmt.Fprintf(&cfg, "public_key=%s\n", hex.EncodeToString(p.NodeKey[:]))
		fmt.Fprintf(&cfg, "endpoint=%s\n", p.NodeKey)
		fmt.Fprintf(&cfg, "allowed_ip=%s/32\n", p.IP)
		if len(p.Endpoints) > 0 {
			e.bind.SetPeerAddr(magicsock.NodeKey(p.NodeKey), p.Endpoints[0])
		}
	}
	return e.dev.IpcSet(cfg.String())
}

func (e *Engine) Close() {
	e.dev.Close()
}
