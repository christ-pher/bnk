package trayui_test

import (
	"testing"

	"github.com/christ-pher/bnk/internal/trayui"
)

func TestParseJoinAcceptsTheServersWindowsCommand(t *testing.T) {
	// Exactly what `bnk-server key new` prints for Windows.
	in := `& ([scriptblock]::Create((irm https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.ps1))) -Server https://203.0.113.7:8443 -Key bnkkey:abc123:def456`
	got, err := trayui.ParseJoin(in)
	if err != nil {
		t.Fatal(err)
	}
	if got.Server != "https://203.0.113.7:8443" {
		t.Errorf("server = %q", got.Server)
	}
	if got.Key != "bnkkey:abc123:def456" {
		t.Errorf("key = %q", got.Key)
	}
}

func TestParseJoinAcceptsTheLinuxCommand(t *testing.T) {
	in := `curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.sh | sudo sh -s -- --server https://203.0.113.7:8443 --key bnkkey:abc123:def456`
	got, err := trayui.ParseJoin(in)
	if err != nil {
		t.Fatal(err)
	}
	if got.Server != "https://203.0.113.7:8443" || got.Key != "bnkkey:abc123:def456" {
		t.Errorf("parsed %+v", got)
	}
}

// The raw key alone is what someone copies from the top of the output.
func TestParseJoinAcceptsABareKeyWithoutAServer(t *testing.T) {
	got, err := trayui.ParseJoin("  bnkkey:abc123:def456  ")
	if err != nil {
		t.Fatal(err)
	}
	if got.Key != "bnkkey:abc123:def456" {
		t.Errorf("key = %q", got.Key)
	}
	if got.Server != "" {
		t.Errorf("server = %q, want empty so the caller supplies it", got.Server)
	}
}

func TestParseJoinFindsAServerAndKeyInAnyOrder(t *testing.T) {
	got, err := trayui.ParseJoin("bnkkey:k1:k2 https://host.example:9999")
	if err != nil {
		t.Fatal(err)
	}
	if got.Server != "https://host.example:9999" || got.Key != "bnkkey:k1:k2" {
		t.Errorf("parsed %+v", got)
	}
}

func TestParseJoinRejectsInputWithoutAKey(t *testing.T) {
	for _, in := range []string{
		"",
		"   ",
		"https://203.0.113.7:8443",
		"just some text the user pasted",
		"bnkkey:",            // no secret or fingerprint
		"bnkkey:onlyonepart", // missing the fingerprint
	} {
		if _, err := trayui.ParseJoin(in); err == nil {
			t.Errorf("ParseJoin(%q) succeeded, want an error", in)
		}
	}
}

// The raw GitHub URL in the pasted command must never be mistaken for
// the control server.
func TestParseJoinIgnoresTheInstallerURL(t *testing.T) {
	in := `curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.sh | sudo sh -s -- --server https://10.0.0.5:8443 --key bnkkey:a:b`
	got, err := trayui.ParseJoin(in)
	if err != nil {
		t.Fatal(err)
	}
	if got.Server != "https://10.0.0.5:8443" {
		t.Errorf("server = %q, want the control server, not the installer URL", got.Server)
	}
}
