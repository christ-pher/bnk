// Package vpnc is the client daemon: state, enrollment, the session/
// reconnect loop, and dataplane assembly. The vpn binary is a thin shell
// around Run; tests inject a netstack TUN factory.
package vpnc

import (
	"context"
	"crypto/tls"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"

	"golang.zx2c4.com/wireguard/tun"

	"vpnmesh/internal/coord"
	"vpnmesh/internal/coord/client"
	"vpnmesh/internal/dataplane"
	"vpnmesh/internal/disco"
	"vpnmesh/internal/netmap"
	"vpnmesh/internal/pin"
)

type Config struct {
	ServerURL string
	EnrollKey string // vpnkey:<secret>:<fingerprint>; required on first run
	StateDir  string
	Hostname  string
	MTU       int // default 1280
	CreateTUN func(prefix netip.Prefix, mtu int) (tun.Device, func() error, error)
	Logf      func(format string, args ...any)

	// EndpointsOverride replaces interface/STUN endpoint discovery with a
	// fixed set. Test hook for simulating hostile NATs; leave nil in
	// production.
	EndpointsOverride []netip.AddrPort
}

// Run brings the node up and blocks until ctx is canceled.
func Run(ctx context.Context, cfg Config) error {
	if cfg.MTU == 0 {
		cfg.MTU = 1280
	}
	if cfg.Logf == nil {
		cfg.Logf = func(string, ...any) {}
	}

	st, haveState, err := loadState(cfg.StateDir)
	if err != nil {
		return err
	}
	if !haveState {
		priv, _, err := generateKeypair()
		if err != nil {
			return err
		}
		st = state{PrivateKey: priv, ServerURL: cfg.ServerURL}
	}
	if st.DiscoPub == (netmap.Key{}) {
		dPriv, dPub, err := disco.NewKeypair()
		if err != nil {
			return err
		}
		st.DiscoPriv, st.DiscoPub = dPriv, dPub
	}
	if cfg.ServerURL != "" {
		st.ServerURL = cfg.ServerURL
	}
	pub, err := publicKey(st.PrivateKey)
	if err != nil {
		return err
	}

	var secret string
	if cfg.EnrollKey != "" {
		s, fp, err := pin.ParseEnrollKey(cfg.EnrollKey)
		if err != nil {
			return err
		}
		secret, st.Fingerprint = s, fp
	}

	var tlsConf *tls.Config
	if strings.HasPrefix(st.ServerURL, "https://") {
		if st.Fingerprint == "" {
			return fmt.Errorf("vpnc: no pinned fingerprint; enrollment key required")
		}
		tlsConf = pin.ClientTLSConfig(st.Fingerprint)
	}
	hc := &http.Client{Transport: &http.Transport{TLSClientConfig: tlsConf}}

	enrolled := st.NodeID != 0 && st.IP.IsValid()
	if !enrolled {
		if secret == "" {
			return fmt.Errorf("vpnc: not enrolled and no enrollment key given")
		}
		resp, err := client.Enroll(ctx, st.ServerURL, hc, coord.EnrollRequest{
			EnrollKey: secret,
			Hostname:  cfg.Hostname,
			OS:        osName(),
			NodeKey:   pub,
			DiscoKey:  st.DiscoPub,
		})
		if err != nil {
			return err
		}
		st.NodeID, st.IP, st.Prefix = resp.NodeID, resp.IP, resp.Prefix
	}
	if err := saveState(cfg.StateDir, st); err != nil {
		return err
	}

	tunDev, cleanup, err := cfg.CreateTUN(netip.PrefixFrom(st.IP, st.Prefix.Bits()), cfg.MTU)
	if err != nil {
		return err
	}
	engine, err := dataplane.New(tunDev, st.PrivateKey, st.DiscoPriv, st.DiscoPub)
	if err != nil {
		return err
	}
	defer engine.Close()
	engine.SetMeshPrefix(st.Prefix)
	defer func() {
		if cleanup != nil {
			cleanup()
		}
	}()

	cache := &netmapCache{}
	if err := serveLocalAPI(ctx, cfg.StateDir, cache, engine, st.ServerURL); err != nil {
		return err
	}

	cfg.Logf("up: node %d, ip %s, wg port %d", st.NodeID, st.IP, engine.LocalPort())
	return sessionLoop(ctx, cfg, st, tlsConf, pub, engine, cache)
}

// sessionLoop keeps a coordination session alive, reapplying netmaps and
// reporting endpoints, with jittered backoff between attempts.
func sessionLoop(ctx context.Context, cfg Config, st state, tlsConf *tls.Config, pub netmap.Key, engine *dataplane.Engine, cache *netmapCache) error {
	backoff := time.Second
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		sess, err := client.Dial(ctx, st.ServerURL, tlsConf, pub, client.Handlers{
			OnNetmap: func(nm netmap.Netmap) {
				cache.set(nm)
				if err := engine.ApplyNetmap(nm); err != nil {
					cfg.Logf("apply netmap: %v", err)
				}
			},
			OnRelayData: func(src netmap.NodeID, pkt []byte) {
				engine.DeliverRelay(uint32(src), pkt)
			},
			OnDiscoFwd: func(src netmap.NodeID, payload []byte) {
				engine.HandleDiscoFwd(payload)
			},
		})
		if err != nil {
			cfg.Logf("coordination dial: %v (retrying in %v)", err, backoff)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(jitter(backoff)):
			}
			if backoff < 30*time.Second {
				backoff *= 2
			}
			continue
		}
		backoff = time.Second
		engine.SetRelaySender(func(dst uint32, pkt []byte) error {
			return sess.SendRelay(netmap.NodeID(dst), pkt)
		})
		engine.SetDiscoFwdSender(func(dst uint32, payload []byte) error {
			return sess.SendDiscoFwd(netmap.NodeID(dst), payload)
		})

		// Advertise local endpoints immediately; the STUN-discovered public
		// endpoint follows asynchronously so a missing responder never
		// stalls the session.
		eps := filterEndpoints(candidateEndpoints(engine.LocalPort()), st.Prefix)
		if cfg.EndpointsOverride != nil {
			eps = cfg.EndpointsOverride
		}
		engine.SetSelfEndpoints(eps)
		if err := sess.SendEndpoints(eps); err != nil {
			cfg.Logf("send endpoints: %v", err)
		}
		go func() {
			if cfg.EndpointsOverride != nil {
				return
			}
			observed, err := stunQuery(ctx, engine, st.ServerURL)
			if err != nil {
				if ctx.Err() == nil {
					cfg.Logf("stun: %v (advertising local endpoints only)", err)
				}
				return
			}
			all := append([]netip.AddrPort{observed}, eps...)
			engine.SetSelfEndpoints(all)
			if err := sess.SendEndpoints(all); err != nil {
				cfg.Logf("send endpoints: %v", err)
			}
		}()

		select {
		case <-ctx.Done():
			sess.Close()
			return ctx.Err()
		case <-sess.Done():
			cfg.Logf("coordination session lost, reconnecting")
		}
	}
}

// stunQuery asks the control server's STUN responder (UDP, same port as
// the TLS listener) for our reflexive address.
func stunQuery(ctx context.Context, engine *dataplane.Engine, serverURL string) (netip.AddrPort, error) {
	u, err := url.Parse(serverURL)
	if err != nil {
		return netip.AddrPort{}, err
	}
	ua, err := net.ResolveUDPAddr("udp4", u.Host)
	if err != nil {
		return netip.AddrPort{}, err
	}
	server, ok := netip.AddrFromSlice(ua.IP)
	if !ok {
		return netip.AddrPort{}, fmt.Errorf("stun: cannot parse resolved addr %v", ua)
	}
	qctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	return engine.QuerySTUN(qctx, netip.AddrPortFrom(server.Unmap(), uint16(ua.Port)))
}

// filterEndpoints drops endpoints inside the mesh prefix: advertising a
// tunnel address makes peers probe through the tunnel itself, proving a
// looping path that then swallows all traffic.
func filterEndpoints(eps []netip.AddrPort, mesh netip.Prefix) []netip.AddrPort {
	if !mesh.IsValid() {
		return eps
	}
	out := eps[:0]
	for _, ep := range eps {
		if !mesh.Contains(ep.Addr()) {
			out = append(out, ep)
		}
	}
	return out
}

// candidateEndpoints pairs every up-interface IPv4 address with the
// WireGuard port, non-loopback addresses first.
func candidateEndpoints(port uint16) []netip.AddrPort {
	var direct, loop []netip.AddrPort
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			ipnet, ok := a.(*net.IPNet)
			if !ok {
				continue
			}
			ip, ok := netip.AddrFromSlice(ipnet.IP)
			if !ok {
				continue
			}
			ip = ip.Unmap()
			if !ip.Is4() {
				continue
			}
			ap := netip.AddrPortFrom(ip, port)
			if ip.IsLoopback() {
				loop = append(loop, ap)
			} else {
				direct = append(direct, ap)
			}
		}
	}
	return append(direct, loop...)
}

func jitter(d time.Duration) time.Duration {
	return d + time.Duration(rand.Int63n(int64(d/2+1)))
}
