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
	"path/filepath"
	"strings"
	"sync"
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
	ServerURL  string
	EnrollKey  string // vpnkey:<secret>:<fingerprint>; required on first run
	StateDir   string
	SocketPath string // local API socket; default <StateDir>/vpn.sock
	Hostname   string
	MTU        int // default 1280
	CreateTUN func(prefix netip.Prefix, mtu int) (tun.Device, func() error, error)
	Logf      func(format string, args ...any)

	// EndpointsOverride replaces interface/STUN endpoint discovery with a
	// fixed set. Test hook for simulating hostile NATs; leave nil in
	// production.
	EndpointsOverride []netip.AddrPort
}

// Run starts the daemon: it serves the local API immediately and keeps
// the tunnel matching the persisted desired state (`vpn up`/`vpn down`)
// until ctx is canceled.
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

	c := &controller{
		cfg:     cfg,
		secret:  secret,
		tlsConf: tlsConf,
		hc:      &http.Client{Transport: &http.Transport{TLSClientConfig: tlsConf}},
		pub:     pub,
		st:      st,
		cache:   &netmapCache{},
		kick:    make(chan struct{}, 1),
	}

	sock := cfg.SocketPath
	if sock == "" {
		sock = filepath.Join(cfg.StateDir, "vpn.sock")
	}
	if err := serveLocalAPI(ctx, sock, c); err != nil {
		return err
	}
	return c.supervise(ctx)
}

// controller owns the tunnel lifecycle. The local API flips its desired
// state; supervise reconciles the tunnel to match.
type controller struct {
	cfg     Config
	secret  string
	tlsConf *tls.Config
	hc      *http.Client
	pub     netmap.Key
	cache   *netmapCache
	kick    chan struct{} // wakes supervise after a desired-state change

	mu         sync.Mutex
	st         state
	engine     *dataplane.Engine  // nil while the tunnel is down
	stopTunnel context.CancelFunc // cancels the running tunnel, if any
}

// supervise runs tunnels while the desired state is up, and idles while
// it is down, until ctx is canceled. A tunnel error while wanted up is
// fatal so the service manager can restart the daemon.
func (c *controller) supervise(ctx context.Context) error {
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if c.wantsUp() {
			tctx, cancel := context.WithCancel(ctx)
			c.mu.Lock()
			c.stopTunnel = cancel
			c.mu.Unlock()
			err := c.runTunnel(tctx)
			cancel()
			c.mu.Lock()
			c.stopTunnel = nil
			c.mu.Unlock()
			if ctx.Err() != nil {
				return err
			}
			if tctx.Err() == nil && err != nil {
				return err
			}
			c.cfg.Logf("down: tunnel torn down")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-c.kick:
		}
	}
}

// runTunnel enrolls if needed, brings the interface up, and runs the
// coordination session until ctx is canceled.
func (c *controller) runTunnel(ctx context.Context) error {
	c.mu.Lock()
	st := c.st
	c.mu.Unlock()

	if st.NodeID == 0 || !st.IP.IsValid() {
		if c.secret == "" {
			return fmt.Errorf("vpnc: not enrolled and no enrollment key given")
		}
		resp, err := client.Enroll(ctx, st.ServerURL, c.hc, coord.EnrollRequest{
			EnrollKey: c.secret,
			Hostname:  c.cfg.Hostname,
			OS:        osName(),
			NodeKey:   c.pub,
			DiscoKey:  st.DiscoPub,
		})
		if err != nil {
			return err
		}
		st.NodeID, st.IP, st.Prefix = resp.NodeID, resp.IP, resp.Prefix
	}
	if err := saveState(c.cfg.StateDir, st); err != nil {
		return err
	}
	c.mu.Lock()
	c.st = st
	c.mu.Unlock()

	tunDev, cleanup, err := c.cfg.CreateTUN(netip.PrefixFrom(st.IP, st.Prefix.Bits()), c.cfg.MTU)
	if err != nil {
		return err
	}
	engine, err := dataplane.New(tunDev, st.PrivateKey, st.DiscoPriv, st.DiscoPub)
	if err != nil {
		if cleanup != nil {
			cleanup()
		}
		return err
	}
	engine.SetMeshPrefix(st.Prefix)
	c.setEngine(engine)
	defer func() {
		c.setEngine(nil)
		engine.Close()
		if cleanup != nil {
			cleanup()
		}
	}()

	c.cfg.Logf("up: node %d, ip %s, wg port %d", st.NodeID, st.IP, engine.LocalPort())
	return sessionLoop(ctx, c.cfg, st, c.tlsConf, c.pub, engine, c.cache)
}

func (c *controller) wantsUp() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.st.Down
}

func (c *controller) setEngine(e *dataplane.Engine) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.engine = e
}

func (c *controller) getEngine() *dataplane.Engine {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.engine
}

func (c *controller) state() state {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.st
}

// setDown persists the desired state and pokes supervise. Going down also
// cancels the running tunnel.
func (c *controller) setDown(down bool) error {
	c.mu.Lock()
	c.st.Down = down
	st := c.st
	stop := c.stopTunnel
	c.mu.Unlock()
	if err := saveState(c.cfg.StateDir, st); err != nil {
		return err
	}
	if down && stop != nil {
		stop()
	}
	select {
	case c.kick <- struct{}{}:
	default:
	}
	return nil
}

// waitEngine polls until the tunnel's engine presence matches want or the
// timeout elapses; it reports whether the state was reached.
func (c *controller) waitEngine(ctx context.Context, want bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if (c.getEngine() != nil) == want {
			return true
		}
		if ctx.Err() != nil || time.Now().After(deadline) {
			return false
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// sessionLoop keeps a coordination session alive, reapplying netmaps and
// reporting endpoints, with jittered backoff between attempts.
func sessionLoop(ctx context.Context, cfg Config, st state, tlsConf *tls.Config, pub netmap.Key, engine *dataplane.Engine, cache *netmapCache) error {
	backoff := time.Second
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		sess, err := client.Dial(ctx, st.ServerURL, tlsConf, st.PrivateKey, client.Handlers{
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
		// Keep the STUN-observed endpoint fresh for the session's lifetime:
		// mappings drift, and peers punch at whatever we last advertised.
		sctx, scancel := context.WithCancel(ctx)
		if cfg.EndpointsOverride == nil {
			go stunRefreshLoop(sctx, 30*time.Second,
				func(qctx context.Context) (netip.AddrPort, error) {
					return stunQuery(qctx, engine, st.ServerURL)
				},
				func(observed netip.AddrPort) {
					all := append([]netip.AddrPort{observed}, eps...)
					engine.SetSelfEndpoints(all)
					if err := sess.SendEndpoints(all); err != nil {
						cfg.Logf("send endpoints: %v", err)
					}
				},
				cfg.Logf,
			)
		}

		select {
		case <-ctx.Done():
			scancel()
			sess.Close()
			return ctx.Err()
		case <-sess.Done():
			scancel()
			cfg.Logf("coordination session lost, reconnecting")
		}
	}
}

// stunRefreshLoop re-queries STUN until ctx ends, invoking onUpdate each
// time the observed address changes (including the first success). NAT
// mappings drift and can be stolen by inbound traffic; a one-shot query
// leaves peers punching at a dead address.
func stunRefreshLoop(ctx context.Context, interval time.Duration, query func(context.Context) (netip.AddrPort, error), onUpdate func(netip.AddrPort), logf func(string, ...any)) {
	var last netip.AddrPort
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		observed, err := query(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			logf("stun: %v (advertising local endpoints only)", err)
		} else if observed != last {
			if last.IsValid() {
				logf("stun: mapping changed %v -> %v", last, observed)
			}
			last = observed
			onUpdate(observed)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
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
