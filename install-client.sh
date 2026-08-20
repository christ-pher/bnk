#!/bin/sh
# install-client.sh — join this Linux machine to the mesh.
#
# Run as root:  sudo ./install-client.sh --server https://VPS_IP:8443 --key bnkkey:...
# (--key can be omitted if this machine enrolled before and still has
#  /var/lib/bnk/client.json)
#
# Also runs straight from the repo with no checkout:
#   curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.sh | sudo sh -s -- --server ... --key ...
#
# Binary source, in order: a prebuilt ./bnk next to this script, a source
# build (repo checkout + Go), else the latest GitHub release for this
# architecture. Installs to /usr/local/bin, config in /etc/bnk, state in
# /var/lib/bnk, systemd unit bnk.service.
set -eu

REPO=christ-pher/bnk
BIN=/usr/local/bin/bnk
CONF_DIR=/etc/bnk
ENV_FILE=$CONF_DIR/bnk.env
UNIT=/etc/systemd/system/bnk.service
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# download_binary <name>: fetch the latest release build for this arch.
download_binary() {
    arch=$(uname -m)
    case "$arch" in
        x86_64) arch=amd64 ;;
        aarch64 | arm64) arch=arm64 ;;
        *) echo "unsupported architecture $arch — build from source: go build ./cmd/$1" >&2; exit 1 ;;
    esac
    url="https://github.com/$REPO/releases/latest/download/$1-linux-$arch"
    echo "downloading $url"
    curl -fSL -o "$BIN" "$url" || {
        echo "download failed — no release published yet? Build from source: go build ./cmd/$1" >&2
        exit 1
    }
    chmod 755 "$BIN"
}

SERVER= KEY=
while [ $# -gt 0 ]; do
    case "$1" in
        --server) SERVER=$2; shift 2 ;;
        --key)    KEY=$2; shift 2 ;;
        *) echo "usage: $0 --server https://host:8443 [--key bnkkey:...]" >&2; exit 1 ;;
    esac
done

[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0 ..." >&2; exit 1; }
[ -n "$SERVER" ] || [ -f "$ENV_FILE" ] || { echo "--server is required on first install" >&2; exit 1; }

if [ -x "$HERE/bnk" ]; then
    install -m 755 "$HERE/bnk" "$BIN"
elif command -v go >/dev/null 2>&1 && [ -f "$HERE/go.mod" ]; then
    echo "building bnk..."
    (cd "$HERE" && CGO_ENABLED=0 go build -o "$BIN" ./cmd/bnk)
else
    download_binary bnk
fi

if [ -n "$SERVER" ]; then
    mkdir -p "$CONF_DIR"
    cat > "$ENV_FILE" <<EOF
BNK_SERVER=$SERVER
BNK_KEY=$KEY
EOF
    chmod 600 "$ENV_FILE"
elif [ -n "$KEY" ]; then
    # Re-enrollment with the stored server: refresh just the key.
    [ -f "$ENV_FILE" ] || { echo "no $ENV_FILE yet — pass --server too" >&2; exit 1; }
    sed -i "s|^BNK_KEY=.*|BNK_KEY=$KEY|" "$ENV_FILE"
fi

cat > "$UNIT" <<'EOF'
[Unit]
Description=bnk client
Documentation=https://github.com/christ-pher/bnk (DEPLOY.md)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# /etc/bnk/bnk.env provides BNK_SERVER (always) and BNK_KEY (only
# until enrolled — the installer blanks it after the first start).
EnvironmentFile=/etc/bnk/bnk.env
ExecStart=/usr/local/bin/bnk run --server ${BNK_SERVER} --key "${BNK_KEY}" --state-dir /var/lib/bnk
Restart=on-failure
RestartSec=2

# Needs root: creates the TUN device and configures routes.
StateDirectory=bnk
# Local API socket lives here so `bnk status` works without sudo.
RuntimeDirectory=bnk
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable bnk
# restart, not `enable --now`: a re-run (upgrade, re-enroll) must replace
# an already-running daemon with the new binary and config.
systemctl restart bnk

echo "waiting for the tunnel..."
i=0
while :; do
    out=$(bnk status 2>/dev/null) || out=""
    case "$out" in
        "") ;; # daemon not answering yet
        "bnk is down"*) bnk up >/dev/null 2>&1 || true ;; # persisted `bnk down` from a past install
        *) break ;;
    esac
    i=$((i + 1))
    if [ "$i" -gt 30 ]; then
        echo "bnk service did not come up after 30s; check: journalctl -u bnk -n 20" >&2
        exit 1
    fi
    sleep 1
done

# The key is single-use and now spent; the node's identity lives in
# /var/lib/bnk. Blank it so the unit never resubmits a dead key.
sed -i 's/^BNK_KEY=.*/BNK_KEY=/' "$ENV_FILE"

echo
bnk status
echo
echo "Done. Everyday commands (no sudo needed for status):"
echo "    bnk status | bnk ping NAME | bnk netcheck    diagnostics"
echo "    bnk down / bnk up                            disconnect / reconnect"
