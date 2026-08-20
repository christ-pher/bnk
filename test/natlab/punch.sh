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
NS=(punch-srv punch-nata punch-natb punch-a punch-b)

# Pre-clean leftovers from any earlier aborted run, so reruns always work.
pkill -f "$WORK/vpnd serve" 2>/dev/null || true
pkill -f "$WORK/vpn up" 2>/dev/null || true
for ns in "${NS[@]}"; do ip netns del "$ns" 2>/dev/null || true; done
ip link del punch-br 2>/dev/null || true
rm -rf "$WORK"
mkdir -p "$WORK"/{srv,a,b}
echo "== building binaries"
go build -o "$WORK/vpnd" ./cmd/vpnd
go build -o "$WORK/vpn" ./cmd/vpn

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
    # The stateful firewall is essential, not decoration: without it,
    # unsolicited inbound packets confirm conntrack entries on the NAT's
    # local stack, stealing the very tuples outbound mappings need
    # (port-preservation fails) — which deadlocks hole punching. Real
    # routers drop unsolicited traffic, so their mappings stay clean.
    ip netns exec "$2" nft -f - <<EOF
table ip nat {
    chain postrouting {
        type nat hook postrouting priority srcnat;
        oifname "wan0" $masq
    }
}
table ip filter {
    chain input {
        type filter hook input priority 0; policy accept;
        iifname "wan0" ct state new,invalid drop
    }
    chain forward {
        type filter hook forward priority 0; policy accept;
        iifname "wan0" ct state new,invalid drop
    }
}
EOF
}
lan_leg punch-a punch-nata 192.168.10
lan_leg punch-b punch-natb 192.168.20

diagnostics() {
    echo "===================== DIAGNOSTICS ====================="
    echo "--- vpnd node ls"
    ip netns exec punch-srv "$WORK/vpnd" node ls --state-dir "$WORK/srv" 2>&1 || true
    for c in a b; do
        echo "--- vpn status ($c)"
        ip netns exec "punch-$c" "$WORK/vpn" status --state-dir "$WORK/$c" 2>&1 || true
        echo "--- vpn netcheck ($c)"
        ip netns exec "punch-$c" "$WORK/vpn" netcheck --state-dir "$WORK/$c" 2>&1 || true
    done
    for n in nata natb; do
        echo "--- nft ruleset ($n)"
        ip netns exec "punch-$n" nft list ruleset 2>&1 || true
        echo "--- conntrack ($n)"
        ip netns exec "punch-$n" conntrack -L 2>/dev/null | head -30 || \
            ip netns exec "punch-$n" cat /proc/net/nf_conntrack 2>/dev/null | head -30 || \
            echo "(conntrack unavailable)"
    done
    echo "--- server sockets"
    ip netns exec punch-srv ss -tunap 2>&1 | head -20 || true
    for f in "$WORK"/srv.log "$WORK"/a.log "$WORK"/b.log; do
        echo "--- $(basename "$f") (last 40 lines)"
        tail -40 "$f" 2>/dev/null || true
    done
    echo "======================================================="
}

echo "== starting vpnd"
ip netns exec punch-srv "$WORK/vpnd" serve --state-dir "$WORK/srv" --listen :8443 \
    >"$WORK/srv.log" 2>&1 &
sleep 1
KEY=$(ip netns exec punch-srv "$WORK/vpnd" key new --reusable --ttl 1h --state-dir "$WORK/srv")

echo "== starting clients (each behind its own NAT)"
ip netns exec punch-a "$WORK/vpn" up --server https://10.99.0.1:8443 --key "$KEY" \
    --state-dir "$WORK/a" --name alpha >"$WORK/a.log" 2>&1 &
ip netns exec punch-b "$WORK/vpn" up --server https://10.99.0.1:8443 --key "$KEY" \
    --state-dir "$WORK/b" --name beta >"$WORK/b.log" 2>&1 &

echo "== waiting for tunnel (relay path first)"
B_IP=""
for i in $(seq 1 30); do
    B_IP=$(python3 -c "import json; print(json.load(open('$WORK/b/client.json'))['ip'])" 2>/dev/null || true)
    [[ -n "$B_IP" ]] && ip netns exec punch-a ping -c1 -W1 "$B_IP" >/dev/null 2>&1 && break
    B_IP=""
    sleep 1
done
if [[ -z "$B_IP" ]]; then
    echo "punch: FAIL - tunnel never came up at all" >&2
    diagnostics
    exit 1
fi
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

# The peer name as seen from client a depends on enrollment order.
PEER=$(ip netns exec punch-a "$WORK/vpn" status --state-dir "$WORK/a" 2>/dev/null | awk 'NR>2 {print $1; exit}')
echo "== vpn ping ${PEER:-?} (twice, with status before/after)"
ip netns exec punch-a "$WORK/vpn" status --state-dir "$WORK/a" || true
ip netns exec punch-a "$WORK/vpn" ping --state-dir "$WORK/a" "$PEER" || true
ip netns exec punch-a "$WORK/vpn" ping --state-dir "$WORK/a" "$PEER" || true
ip netns exec punch-a "$WORK/vpn" status --state-dir "$WORK/a" || true

echo "== netcheck (alpha side): path-manager state and relay counters"
NETCHECK=$(ip netns exec punch-a "$WORK/vpn" netcheck --state-dir "$WORK/a" 2>&1 || true)
echo "$NETCHECK"
echo "== netcheck (beta side)"
ip netns exec punch-b "$WORK/vpn" netcheck --state-dir "$WORK/b" 2>&1 || true
BEST=$(echo "$NETCHECK" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(iter(d.get("peers",{}).values()),{}).get("best",""))' 2>/dev/null || true)

if [[ "$MODE" == "symmetric" ]]; then
    if [[ "$DIRECT" == "yes" ]]; then
        echo "punch: FAIL - direct path ($BEST) under symmetric NATs is impossible; a bogus candidate was proven" >&2
        diagnostics
        exit 1
    fi
    echo "== PASS: symmetric NATs correctly fell back to relay, tunnel works"
    exit 0
fi
if [[ "$DIRECT" == "yes" ]]; then
    # The proven address must be a real underlay (simulated-internet) one.
    if [[ "$BEST" == 10.99.* ]]; then
        echo "== PASS: hole punch succeeded, direct via real underlay address $BEST"
        exit 0
    fi
    echo "punch: FAIL - 'direct' via $BEST, which is not an underlay address (loop or LAN)" >&2
    diagnostics
    exit 1
fi
echo "punch: FAIL - never upgraded to a direct path (still relayed)" >&2
diagnostics
exit 1
