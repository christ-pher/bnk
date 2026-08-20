#!/bin/sh
# purge-old-vpnmesh.sh — remove everything installed under the OLD naming
# (vpn / vpnd / vpnmesh), from before the rename to bnk. TEMPORARY helper
# for migrating existing machines; safe to run on server or client, and
# harmless to run twice. Touches nothing named bnk.
#
#   curl -fsSL https://raw.githubusercontent.com/christ-pher/bnk/main/purge-old-vpnmesh.sh | sudo sh
#
# THIS DELETES the old node identity (/var/lib/vpn) and server state
# (/var/lib/vpnd, including its TLS cert and node registry). That is the
# point — machines re-enroll fresh as bnk — but there is no undo.
set -u

[ "$(id -u)" = 0 ] || { echo "run as root: sudo $0" >&2; exit 1; }

# Old services: stop, disable, delete the units.
for svc in vpn vpnd; do
    if [ -f "/etc/systemd/system/$svc.service" ]; then
        echo "removing service $svc"
        systemctl stop "$svc" 2>/dev/null
        systemctl disable "$svc" 2>/dev/null
        rm -f "/etc/systemd/system/$svc.service"
    fi
done
systemctl daemon-reload

# Old binaries, state, config, runtime socket dir.
for path in \
    /usr/local/bin/vpn \
    /usr/local/bin/vpnd \
    /var/lib/vpn \
    /var/lib/vpnd \
    /etc/vpnmesh \
    /run/vpnmesh; do
    if [ -e "$path" ]; then
        echo "removing $path"
        rm -rf "$path"
    fi
done

echo
echo "done — all old vpnmesh-era files are gone."
echo "install bnk fresh: see DEPLOY.md (server one-liner + the command bnk-server key new prints)"
