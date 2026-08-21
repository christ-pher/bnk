package vpnc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/christ-pher/bnk/internal/dataplane"
	"github.com/christ-pher/bnk/internal/magicsock"
	"github.com/christ-pher/bnk/internal/netmap"
)

// netmapCache holds the latest pushed netmap for status reporting.
type netmapCache struct {
	mu sync.Mutex
	nm netmap.Netmap
}

func (c *netmapCache) set(nm netmap.Netmap) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nm = nm
}

func (c *netmapCache) get() netmap.Netmap {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.nm
}

// The local API is split along privilege lines so each OS can enforce
// the same policy with its own mechanism: diagnostics are readable by
// any local user, control verbs are not. Linux mounts both sets on one
// unix socket and gates control with SO_PEERCRED; Windows mounts them on
// two named pipes whose ACLs do the gating.

// registerDiagnostics adds the read-only routes: they expose no secrets
// and are safe for any local user.
func registerDiagnostics(mux *http.ServeMux, c *controller) {
	// withEngine gates handlers that need a live tunnel.
	withEngine := func(h func(w http.ResponseWriter, r *http.Request, engine *dataplane.Engine)) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			engine := c.getEngine()
			if engine == nil {
				http.Error(w, "bnk is down (run `bnk up` to connect)", http.StatusServiceUnavailable)
				return
			}
			h(w, r, engine)
		}
	}

	mux.HandleFunc("GET /status", func(w http.ResponseWriter, r *http.Request) {
		engine := c.getEngine()
		if engine == nil {
			st := c.state()
			json.NewEncoder(w).Encode(Status{
				Enrolled: c.enrolled(),
				Self:     SelfStatus{ID: st.NodeID, Name: c.cfg.Hostname, IP: st.IP},
			})
			return
		}
		out := buildStatus(c.cache.get(), c.cfg.Hostname, engine.PeerPaths())
		out.Running = true
		out.Enrolled = true
		json.NewEncoder(w).Encode(out)
	})
	mux.HandleFunc("GET /ping", withEngine(func(w http.ResponseWriter, r *http.Request, engine *dataplane.Engine) {
		name := r.URL.Query().Get("peer")
		var key magicsock.NodeKey
		found := false
		for _, p := range c.cache.get().Peers {
			if p.Name == name {
				key, found = magicsock.NodeKey(p.NodeKey), true
			}
		}
		if !found {
			http.Error(w, "unknown peer "+name, http.StatusNotFound)
			return
		}
		res, err := engine.PingPeer(key, 5*time.Second)
		if err != nil {
			// Append the path snapshot: it distinguishes "network dead"
			// (stale pong) from "pongs flow but Ping is broken" (fresh).
			if d, ok := engine.PeerDebug(key); ok {
				if !d.HasPong {
					err = fmt.Errorf("no direct path to this peer has ever been proven — it is likely reachable only via the relay (normal behind port-randomizing NATs). Underlying: %w [candidates=%v]", err, d.Candidates)
				} else {
					err = fmt.Errorf("%w [best=%v lastPong=%v ago lastPing=%v ago candidates=%v]",
						err, d.Best, d.LastPongAge.Round(time.Millisecond), d.LastPingAge.Round(time.Millisecond), d.Candidates)
				}
			}
			http.Error(w, err.Error(), http.StatusGatewayTimeout)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"addr":   res.Addr.String(),
			"rtt_ms": float64(res.RTT.Microseconds()) / 1000,
		})
	}))
	mux.HandleFunc("GET /netcheck", withEngine(func(w http.ResponseWriter, r *http.Request, engine *dataplane.Engine) {
		tx, rx := engine.RelayStats()
		peers := map[string]any{}
		for _, p := range c.cache.get().Peers {
			if d, ok := engine.PeerDebug(magicsock.NodeKey(p.NodeKey)); ok {
				lastPong := "never"
				if d.HasPong {
					lastPong = d.LastPongAge.Round(time.Millisecond).String()
				}
				peers[p.Name] = map[string]any{
					"best":          d.Best.String(),
					"last_pong_ago": lastPong,
					"last_ping_ago": d.LastPingAge.Round(time.Millisecond).String(),
					"candidates":    d.Candidates,
				}
			}
		}
		out := map[string]any{
			"local_endpoints": candidateEndpoints(engine.LocalPort()),
			"relay_tx":        tx,
			"relay_rx":        rx,
			"peers":           peers,
			"disco_events":    engine.DiscoEvents(),
		}
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		if observed, err := stunQuery(ctx, engine, c.state().ServerURL); err == nil {
			out["stun_observed"] = observed.String()
		} else {
			out["stun_error"] = err.Error()
		}
		json.NewEncoder(w).Encode(out)
	}))
}

// registerControl adds the routes that change the tunnel's desired
// state. gate wraps each handler with the platform's authorization check
// (nil means the transport itself already restricted access).
func registerControl(mux *http.ServeMux, c *controller, gate func(http.HandlerFunc) http.HandlerFunc) {
	if gate == nil {
		gate = func(h http.HandlerFunc) http.HandlerFunc { return h }
	}
	mux.HandleFunc("POST /up", gate(func(w http.ResponseWriter, r *http.Request) {
		if err := c.setDown(false); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !c.waitEngine(r.Context(), true, 15*time.Second) {
			http.Error(w, "still connecting; check the daemon logs", http.StatusGatewayTimeout)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"running": true})
	}))
	// Joining is a control verb: it decides which mesh this machine
	// belongs to, so it is gated exactly like up and down.
	mux.HandleFunc("POST /join", gate(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Server string `json:"server"`
			Key    string `json:"key"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := c.join(req.Server, req.Key); err != nil {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		if !c.waitEngine(r.Context(), true, 30*time.Second) {
			http.Error(w, "signed in, but the tunnel has not come up yet — check the daemon logs", http.StatusGatewayTimeout)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"running": true})
	}))
	mux.HandleFunc("POST /down", gate(func(w http.ResponseWriter, r *http.Request) {
		if err := c.setDown(true); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !c.waitEngine(r.Context(), false, 15*time.Second) {
			http.Error(w, "tunnel did not shut down in time", http.StatusGatewayTimeout)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"running": false})
	}))
}

func buildStatus(nm netmap.Netmap, selfName string, paths []dataplane.PeerPath) Status {
	byKey := make(map[magicsock.NodeKey]dataplane.PeerPath, len(paths))
	for _, p := range paths {
		byKey[p.Key] = p
	}
	st := Status{Self: SelfStatus{ID: nm.SelfID, Name: selfName, IP: nm.SelfIP.Addr()}}
	for _, p := range nm.Peers {
		ps := PeerStatus{ID: p.ID, Name: p.Name, IP: p.IP, Online: p.Online, Path: "relay"}
		if pp, ok := byKey[magicsock.NodeKey(p.NodeKey)]; ok {
			if pp.Direct {
				ps.Path = "direct"
			}
			ps.LastHandshake = pp.LastHandshake
		}
		st.Peers = append(st.Peers, ps)
	}
	return st
}
