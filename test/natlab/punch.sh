#!/usr/bin/env bash
# NAT traversal test: two clients behind separate masquerading NATs
# (port-restricted cone, the common home-router case) must first connect
# via the relay, then hole-punch to a direct path.
#
# Requires root: sudo test/natlab/punch.sh
# Pass "symmetric" to make both NATs fully-random (expected outcome: the
# tunnel still works but stays on the relay).
#
#   [a]--[natA]--\
#                 [wan 10.99.0.0/24]--[srv: vpnd + STUN]
#   [b]--[natB]--/
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-cone}"
if [[ $EUID -ne 0 ]]; then
    echo "punch: must run as root (sudo $0)" >&2
    exit 1
fi

WORK=/tmp/vpnmesh-natlab
rm -rf "$WORK"
mkdir -p "$WORK"/{srv,a,b}
echo "== building binaries"
go build -o "$WORK/vpnd" ./cmd/vpnd
go build -o "$WORK/vpn" ./cmd/vpn

NS=(punch-srv punch-nata punch-natb punch-a punch-b)
cleanup() {
    set +e
    pkill -f "$WORK/vpnd serve" 2>/dev/null
    pkill -f "$WORK/vpn up" 2>/dev/null
    for ns in "${NS[@]}"; do ip netns del "$ns" 2>/dev/null; done
    ip link del punch-br 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "== building topology ($MODE NATs)"
ip link add punch-br type bridge
ip link set punch-br up

for ns in "${NS[@]}"; do ip netns add "$ns"; ip netns exec "$ns" ip link set lo up; done

# WAN-facing veths onto the bridge.
wan_iface() { # ns wan_ip
    ip link add "veth-$1" type veth peer name wan0 netns "$1"
    ip link set "veth-$1" master punch-br up
    ip netns exec "$1" ip addr add "$2/24" dev wan0
    ip netns exec "$1" ip link set wan0 up
}
wan_iface punch-srv  10.99.0.1
wan_iface punch-nata 10.99.0.11
wan_iface punch-natb 10.99.0.12

# LAN legs: client <-> its NAT.
lan_leg() { # client_ns nat_ns subnet_prefix
    ip link add lan0 netns "$2" type veth peer name eth0 netns "$1"
    ip netns exec "$2" ip addr add "$3.1/24" dev lan0
    ip netns exec "$2" ip link set lan0 up
    ip netns exec "$1" ip addr add "$3.2/24" dev eth0
    ip netns exec "$1" ip link set eth0 up
    ip netns exec "$1" ip route add default via "$3.1"
    ip netns exec "$2" sysctl -qw net.ipv4.ip_forward=1
    local masq="masquerade"
    [[ "$MODE" == "symmetric" ]] && masq="masquerade fully-random"
    ip netns exec "$2" nft -f - <<EOF
table ip nat {
    chain postrouting {
        type nat hook postrouting priority srcnat;
        oifname "wan0" $masq
    }
}
EOF
}
lan_leg punch-a punch-nata 192.168.10
lan_leg punch-b punch-natb 192.168.20

echo "== starting vpnd"
ip netns exec punch-srv "$WORK/vpnd" serve --state-dir "$WORK/srv" --listen :8443 &
sleep 1
KEY=$(ip netns exec punch-srv "$WORK/vpnd" key new --state-dir "$WORK/srv")

echo "== starting clients (each behind its own NAT)"
ip netns exec punch-a "$WORK/vpn" up --server https://10.99.0.1:8443 --key "$KEY" \
    --state-dir "$WORK/a" --name alpha &
ip netns exec punch-b "$WORK/vpn" up --server https://10.99.0.1:8443 --key "$KEY" \
    --state-dir "$WORK/b" --name beta &

echo "== waiting for tunnel (relay path first)"
B_IP=""
for i in $(seq 1 30); do
    B_IP=$(python3 -c "import json; print(json.load(open('$WORK/b/client.json'))['ip'])" 2>/dev/null || true)
    [[ -n "$B_IP" ]] && ip netns exec punch-a ping -c1 -W1 "$B_IP" >/dev/null 2>&1 && break
    B_IP=""
    sleep 1
done
[[ -n "$B_IP" ]] || { echo "punch: FAIL - tunnel never came up at all" >&2; exit 1; }
echo "== tunnel is up (alpha pinged beta at $B_IP)"

echo "== waiting for direct path"
DIRECT=no
for i in $(seq 1 60); do
    if ip netns exec punch-a "$WORK/vpn" status --state-dir "$WORK/a" 2>/dev/null | grep -q direct; then
        DIRECT=yes
        break
    fi
    sleep 1
done

ip netns exec punch-a "$WORK/vpn" status --state-dir "$WORK/a" || true
ip netns exec punch-a "$WORK/vpn" ping --state-dir "$WORK/a" beta || true

if [[ "$MODE" == "symmetric" ]]; then
    echo "== symmetric mode: relay is the expected outcome; tunnel works = PASS"
    exit 0
fi
if [[ "$DIRECT" == "yes" ]]; then
    echo "== PASS: hole punch succeeded, path is direct through both NATs"
    exit 0
fi
echo "punch: FAIL - never upgraded to a direct path (still relayed)" >&2
exit 1
