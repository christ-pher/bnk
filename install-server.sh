#!/bin/sh
# install-server.sh — set up the vpnmesh control server on this machine.
#
# Run as root on the VPS:   sudo ./install-server.sh
#
# Uses a prebuilt ./vpnd next to this script if present, otherwise builds
# it (needs Go). Installs to /usr/local/bin, state in /var/lib/vpnd,
# systemd unit vpnd.service listening on :8443 (tcp+udp).
set -eu

BIN=/usr/local/bin/vpnd
STATE_DIR=/var/lib/vpnd
UNIT=/etc/systemd/system/vpnd.service
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0" >&2; exit 1; }

if [ -x "$HERE/vpnd" ]; then
    install -m 755 "$HERE/vpnd" "$BIN"
elif command -v go >/dev/null 2>&1 && [ -f "$HERE/go.mod" ]; then
    echo "building vpnd..."
    (cd "$HERE" && CGO_ENABLED=0 go build -o "$BIN" ./cmd/vpnd)
else
    echo "no ./vpnd binary next to this script and no Go toolchain to build one" >&2
    echo "build elsewhere with: CGO_ENABLED=0 go build -o vpnd ./cmd/vpnd — then scp both files here" >&2
    exit 1
fi

cat > "$UNIT" <<'EOF'
[Unit]
Description=vpnmesh control server
Documentation=https://github.com/chris/vpnmesh (DEPLOY.md)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/vpnd serve --state-dir /var/lib/vpnd --listen :8443
Restart=on-failure
RestartSec=2

# State dir holds the TLS key, node registry, and admin socket.
StateDirectory=vpnd
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/vpnd
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vpnd
# restart, not `enable --now`: a re-run must replace a running daemon.
systemctl restart vpnd

# The fingerprint proves the server identity to enrolling clients; it is
# printed on every start.
sleep 1
echo
echo "vpnd installed and running."
journalctl -u vpnd -n 3 --no-pager | grep -o "cert fingerprint.*" || true
echo
echo "Open port 8443 (tcp AND udp) in your firewall, then mint one key per client:"
echo "    vpnd key new"
echo "Control it later with: vpnd up / vpnd down"
