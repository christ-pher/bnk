#!/bin/sh
# install-client.sh — join this Linux machine to the mesh.
#
# Run as root:  sudo ./install-client.sh --server https://VPS_IP:8443 --key vpnkey:...
# (--key can be omitted if this machine enrolled before and still has
#  /var/lib/vpn/client.json)
#
# Uses a prebuilt ./vpn next to this script if present, otherwise builds
# it (needs Go). Installs to /usr/local/bin, config in /etc/vpnmesh,
# state in /var/lib/vpn, systemd unit vpn.service.
set -eu

BIN=/usr/local/bin/vpn
CONF_DIR=/etc/vpnmesh
ENV_FILE=$CONF_DIR/vpn.env
UNIT=/etc/systemd/system/vpn.service
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

SERVER= KEY=
while [ $# -gt 0 ]; do
    case "$1" in
        --server) SERVER=$2; shift 2 ;;
        --key)    KEY=$2; shift 2 ;;
        *) echo "usage: $0 --server https://host:8443 [--key vpnkey:...]" >&2; exit 1 ;;
    esac
done

[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0 ..." >&2; exit 1; }
[ -n "$SERVER" ] || [ -f "$ENV_FILE" ] || { echo "--server is required on first install" >&2; exit 1; }

if [ -x "$HERE/vpn" ]; then
    install -m 755 "$HERE/vpn" "$BIN"
elif command -v go >/dev/null 2>&1 && [ -f "$HERE/go.mod" ]; then
    echo "building vpn..."
    (cd "$HERE" && CGO_ENABLED=0 go build -o "$BIN" ./cmd/vpn)
else
    echo "no ./vpn binary next to this script and no Go toolchain to build one" >&2
    echo "build elsewhere with: CGO_ENABLED=0 go build -o vpn ./cmd/vpn — then scp both files here" >&2
    exit 1
fi

if [ -n "$SERVER" ]; then
    mkdir -p "$CONF_DIR"
    cat > "$ENV_FILE" <<EOF
VPN_SERVER=$SERVER
VPN_KEY=$KEY
EOF
    chmod 600 "$ENV_FILE"
fi

cat > "$UNIT" <<'EOF'
[Unit]
Description=vpnmesh client
Documentation=https://github.com/chris/vpnmesh (DEPLOY.md)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# /etc/vpnmesh/vpn.env provides VPN_SERVER (always) and VPN_KEY (only
# until enrolled — the installer blanks it after the first start).
EnvironmentFile=/etc/vpnmesh/vpn.env
ExecStart=/usr/local/bin/vpn run --server ${VPN_SERVER} --key "${VPN_KEY}" --state-dir /var/lib/vpn
Restart=on-failure
RestartSec=2

# Needs root: creates the TUN device and configures routes.
StateDirectory=vpn
# Local API socket lives here so `vpn status` works without sudo.
RuntimeDirectory=vpnmesh
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vpn

echo "waiting for the tunnel..."
i=0
until out=$(vpn status 2>/dev/null) && case "$out" in "vpn is down"*) false ;; *) true ;; esac; do
    i=$((i + 1))
    if [ "$i" -gt 30 ]; then
        echo "vpn service did not come up after 30s; check: journalctl -u vpn -n 20" >&2
        exit 1
    fi
    sleep 1
done

# The key is single-use and now spent; the node's identity lives in
# /var/lib/vpn. Blank it so the unit never resubmits a dead key.
sed -i 's/^VPN_KEY=.*/VPN_KEY=/' "$ENV_FILE"

echo
vpn status
echo
echo "Done. Everyday commands (no sudo needed for status):"
echo "    vpn status | vpn ping NAME | vpn netcheck    diagnostics"
echo "    vpn down / vpn up                            disconnect / reconnect"
