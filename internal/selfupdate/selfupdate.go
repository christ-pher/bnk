// Package selfupdate replaces the running binary with the latest GitHub
// release build, verifying it against the release's SHA256SUMS. State on
// disk is untouched: an update is download, swap, restart.
package selfupdate

import (
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type Config struct {
	BaseURL string // repo URL, e.g. https://github.com/christ-pher/bnk
	Asset   string // release asset stem: "bnk" or "bnk-server"
	Version string // currently running version tag ("dev" for local builds)
	Target  string // binary path to replace; default: the running executable
	Service string // systemd unit to restart afterwards; "" skips
	Out     io.Writer
}

// Run updates the binary in place if a newer release exists.
func Run(cfg Config) error {
	if cfg.Out == nil {
		cfg.Out = os.Stdout
	}
	if cfg.Target == "" {
		exe, err := os.Executable()
		if err != nil {
			return err
		}
		cfg.Target = exe
	}

	tag, err := latestTag(cfg.BaseURL)
	if err != nil {
		return err
	}
	if tag == cfg.Version {
		fmt.Fprintf(cfg.Out, "already up to date (%s)\n", tag)
		return nil
	}

	asset := fmt.Sprintf("%s-linux-%s", cfg.Asset, runtime.GOARCH)
	base := fmt.Sprintf("%s/releases/download/%s/", cfg.BaseURL, tag)
	fmt.Fprintf(cfg.Out, "downloading %s %s...\n", asset, tag)
	bin, err := fetch(base + asset)
	if err != nil {
		return err
	}
	sums, err := fetch(base + "SHA256SUMS")
	if err != nil {
		return err
	}
	if err := verify(bin, sums, asset); err != nil {
		return err
	}

	// Write beside the target and rename over it: atomic on Linux, and
	// safe while the old binary is still executing.
	tmp, err := os.CreateTemp(filepath.Dir(cfg.Target), "."+cfg.Asset+"-*.new")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(bin); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o755); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), cfg.Target); err != nil {
		return err
	}
	fmt.Fprintf(cfg.Out, "updated %s -> %s\n", cfg.Version, tag)

	if cfg.Service != "" {
		if err := exec.Command("systemctl", "restart", cfg.Service).Run(); err != nil {
			fmt.Fprintf(cfg.Out, "restart the service to finish: systemctl restart %s (%v)\n", cfg.Service, err)
		} else {
			fmt.Fprintf(cfg.Out, "%s restarted\n", cfg.Service)
		}
	}
	return nil
}

// LatestTag resolves the tag GitHub's releases/latest redirect points
// at. It is exported because the tray checks for updates without being
// able to install one: replacing files under Program Files needs
// privileges the tray deliberately does not have.
func LatestTag(baseURL string) (string, error) { return latestTag(baseURL) }

// latestTag resolves the tag GitHub's releases/latest redirect points at.
func latestTag(baseURL string) (string, error) {
	hc := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := hc.Get(baseURL + "/releases/latest")
	if err != nil {
		return "", err
	}
	resp.Body.Close()
	loc := resp.Header.Get("Location")
	i := strings.LastIndex(loc, "/tag/")
	if i < 0 {
		return "", fmt.Errorf("no release found (releases/latest answered %d %q)", resp.StatusCode, loc)
	}
	return loc[i+len("/tag/"):], nil
}

func fetch(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	return io.ReadAll(resp.Body)
}

// verify checks bin against the `<hex>  <name>` line for asset in sums.
func verify(bin, sums []byte, asset string) error {
	got := fmt.Sprintf("%x", sha256.Sum256(bin))
	for _, line := range strings.Split(string(sums), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && f[1] == asset {
			if f[0] != got {
				return fmt.Errorf("checksum mismatch for %s: got %s, want %s", asset, got, f[0])
			}
			return nil
		}
	}
	return fmt.Errorf("checksum for %s not found in SHA256SUMS", asset)
}
