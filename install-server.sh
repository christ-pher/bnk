#!/bin/sh
# install-server.sh — set up the bnk control server on this machine.
#
# Run as root on the VPS:   sudo ./install-server.sh
#
# Uses a prebuilt ./bnk-server next to this script if present, otherwise builds
# it (needs Go). Installs to /usr/local/bin, state in /var/lib/bnk-server,
# systemd unit bnk-server.service listening on :8443 (tcp+udp).
set -eu

BIN=/usr/local/bin/bnk-server
STATE_DIR=/var/lib/bnk-server
UNIT=/etc/systemd/system/bnk-server.service
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0" >&2; exit 1; }

if [ -x "$HERE/bnk-server" ]; then
    install -m 755 "$HERE/bnk-server" "$BIN"
elif command -v go >/dev/null 2>&1 && [ -f "$HERE/go.mod" ]; then
    echo "building bnk-server..."
    (cd "$HERE" && CGO_ENABLED=0 go build -o "$BIN" ./cmd/bnk-server)
else
    echo "no ./bnk-server binary next to this script and no Go toolchain to build one" >&2
    echo "build elsewhere with: CGO_ENABLED=0 go build -o bnk-server ./cmd/bnk-server — then scp both files here" >&2
    exit 1
fi

cat > "$UNIT" <<'EOF'
[Unit]
Description=bnk control server
Documentation=https://github.com/christ-pher/bnk (DEPLOY.md)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/bnk-server serve --state-dir /var/lib/bnk-server --listen :8443
Restart=on-failure
RestartSec=2

# State dir holds the TLS key, node registry, and admin socket.
StateDirectory=bnk-server
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/bnk-server
ProtectHome=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable bnk-server
# restart, not `enable --now`: a re-run must replace a running daemon.
systemctl restart bnk-server

# The fingerprint proves the server identity to enrolling clients; it is
# printed on every start.
sleep 1
echo
echo "bnk-server installed and running."
journalctl -u bnk-server -n 3 --no-pager | grep -o "cert fingerprint.*" || true
echo
echo "Open port 8443 (tcp AND udp) in your firewall, then mint one key per client:"
echo "    bnk-server key new"
echo "Control it later with: bnk-server up / bnk-server down"
