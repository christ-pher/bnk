package vpnc

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"vpnmesh/internal/dataplane"
	"vpnmesh/internal/magicsock"
	"vpnmesh/internal/netmap"
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

// serveLocalAPI exposes daemon state to the CLI over a unix socket in the
// state directory. It shuts down when ctx is canceled.
func serveLocalAPI(ctx context.Context, stateDir string, cache *netmapCache, engine *dataplane.Engine, serverURL string) error {
	sock := filepath.Join(stateDir, "vpn.sock")
	os.Remove(sock)
	ln, err := net.Listen("unix", sock)
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /status", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(buildStatus(cache.get(), engine.PeerPaths()))
	})
	mux.HandleFunc("GET /ping", func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Query().Get("peer")
		var key magicsock.NodeKey
		found := false
		for _, p := range cache.get().Peers {
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
				err = fmt.Errorf("%w [best=%v lastPong=%v ago lastPing=%v ago candidates=%v]",
					err, d.Best, d.LastPongAge.Round(time.Millisecond), d.LastPingAge.Round(time.Millisecond), d.Candidates)
			}
			http.Error(w, err.Error(), http.StatusGatewayTimeout)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"addr":   res.Addr.String(),
			"rtt_ms": float64(res.RTT.Microseconds()) / 1000,
		})
	})
	mux.HandleFunc("GET /netcheck", func(w http.ResponseWriter, r *http.Request) {
		tx, rx := engine.RelayStats()
		peers := map[string]any{}
		for _, p := range cache.get().Peers {
			if d, ok := engine.PeerDebug(magicsock.NodeKey(p.NodeKey)); ok {
				peers[p.Name] = map[string]any{
					"best":          d.Best.String(),
					"last_pong_ago": d.LastPongAge.Round(time.Millisecond).String(),
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
		}
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		if observed, err := stunQuery(ctx, engine, serverURL); err == nil {
			out["stun_observed"] = observed.String()
		} else {
			out["stun_error"] = err.Error()
		}
		json.NewEncoder(w).Encode(out)
	})
	go func() {
		<-ctx.Done()
		ln.Close()
	}()
	go http.Serve(ln, mux)
	return nil
}

func buildStatus(nm netmap.Netmap, paths []dataplane.PeerPath) Status {
	byKey := make(map[magicsock.NodeKey]dataplane.PeerPath, len(paths))
	for _, p := range paths {
		byKey[p.Key] = p
	}
	st := Status{Self: SelfStatus{ID: nm.SelfID, IP: nm.SelfIP.Addr()}}
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
