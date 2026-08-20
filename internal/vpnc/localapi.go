package vpnc

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"

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
func serveLocalAPI(ctx context.Context, stateDir string, cache *netmapCache, engine *dataplane.Engine) error {
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
