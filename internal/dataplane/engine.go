// Package dataplane assembles the client's packet path: TUN device,
// (later) ACL filter, WireGuard device, and magicsock Bind.
package dataplane

import (
	"encoding/hex"
	"fmt"
	"net/netip"
	"strings"
	"sync"
	"time"

	"golang.zx2c4.com/wireguard/device"
	"golang.zx2c4.com/wireguard/tun"

	"vpnmesh/internal/magicsock"
	"vpnmesh/internal/netmap"
)

// fallbackGrace is how long a never-handshaked peer may hold a direct
// address before being demoted to the relay path.
const fallbackGrace = 5 * time.Second

type Engine struct {
	mu          sync.Mutex
	bind        *magicsock.Bind
	dev         *device.Device
	applied     map[magicsock.NodeKey]netip.AddrPort // direct addr currently set
	directSince map[magicsock.NodeKey]time.Time      // unproven direct paths
	done        chan struct{}
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
	e := &Engine{
		bind:        bind,
		dev:         dev,
		applied:     make(map[magicsock.NodeKey]netip.AddrPort),
		directSince: make(map[magicsock.NodeKey]time.Time),
		done:        make(chan struct{}),
	}
	go e.watchdog()
	return e, nil
}

// watchdog demotes direct paths that never produce a handshake, so a peer
// advertising an unreachable endpoint ends up on the relay instead of
// staying dark. Phase 4's path manager replaces this heuristic.
func (e *Engine) watchdog() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-e.done:
			return
		case <-ticker.C:
		}
		shaken := e.handshakedPeers()
		e.mu.Lock()
		for key, since := range e.directSince {
			if shaken[key] {
				delete(e.directSince, key) // proven; stop watching
				continue
			}
			if time.Since(since) > fallbackGrace {
				e.bind.ClearPeerAddr(key)
				delete(e.directSince, key)
				delete(e.applied, key)
			}
		}
		e.mu.Unlock()
	}
}

// handshakedPeers reports which peers have ever completed a handshake.
func (e *Engine) handshakedPeers() map[magicsock.NodeKey]bool {
	out := make(map[magicsock.NodeKey]bool)
	raw, err := e.dev.IpcGet()
	if err != nil {
		return out
	}
	var current magicsock.NodeKey
	var haveCurrent bool
	for _, line := range strings.Split(raw, "\n") {
		if v, ok := strings.CutPrefix(line, "public_key="); ok {
			b, err := hex.DecodeString(v)
			haveCurrent = err == nil && len(b) == 32
			if haveCurrent {
				copy(current[:], b)
			}
			continue
		}
		if v, ok := strings.CutPrefix(line, "last_handshake_time_sec="); ok && haveCurrent && v != "0" {
			out[current] = true
		}
	}
	return out
}

func (e *Engine) LocalPort() uint16 {
	return e.bind.LocalPort()
}

// SetRelaySender wires the coordination session's relay transport into the
// packet path.
func (e *Engine) SetRelaySender(send func(dst uint32, pkt []byte) error) {
	e.bind.SetRelaySender(send)
}

// DeliverRelay feeds a relayed WireGuard packet from the coordination
// session into the packet path.
func (e *Engine) DeliverRelay(src uint32, pkt []byte) {
	e.bind.DeliverRelay(src, pkt)
}

// ApplyNetmap reconfigures the device and path table to match nm. Peers
// absent from nm are removed (replace_peers); each peer's identity is its
// node key, and its freshest known endpoint feeds the Bind's path table.
func (e *Engine) ApplyNetmap(nm netmap.Netmap) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	var cfg strings.Builder
	cfg.WriteString("replace_peers=true\n")
	inMap := make(map[magicsock.NodeKey]bool, len(nm.Peers))
	for _, p := range nm.Peers {
		key := magicsock.NodeKey(p.NodeKey)
		inMap[key] = true
		fmt.Fprintf(&cfg, "public_key=%s\n", hex.EncodeToString(p.NodeKey[:]))
		fmt.Fprintf(&cfg, "endpoint=%s\n", p.NodeKey)
		fmt.Fprintf(&cfg, "allowed_ip=%s/32\n", p.IP)
		e.bind.SetPeerRelay(key, uint32(p.ID))
		if len(p.Endpoints) > 0 && e.applied[key] != p.Endpoints[0] {
			e.bind.SetPeerAddr(key, p.Endpoints[0])
			e.applied[key] = p.Endpoints[0]
			e.directSince[key] = time.Now()
		}
	}
	for key := range e.applied {
		if !inMap[key] {
			e.bind.ClearPeerAddr(key)
			delete(e.applied, key)
			delete(e.directSince, key)
		}
	}
	return e.dev.IpcSet(cfg.String())
}

func (e *Engine) Close() {
	close(e.done)
	e.dev.Close()
}
