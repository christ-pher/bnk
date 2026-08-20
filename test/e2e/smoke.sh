#!/usr/bin/env bash
# End-to-end smoke test with real binaries, real TUNs, and network
# namespaces. Requires root: sudo test/e2e/smoke.sh
#
# Topology: three namespaces (srv, a, b) joined by a bridge in a lab
# namespace, so nothing touches the host's network config.
#
#   [a]-----\
#            [lab bridge 10.99.0.0/24]----[srv: bnk-server]
#   [b]-----/
#
# Asserts: both clients enroll, and a pings b over tunnel IPs.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $EUID -ne 0 ]]; then
    echo "smoke: must run as root (sudo $0)" >&2
    exit 1
fi

WORK=/tmp/bnk-e2e
NS=(bnklab-srv bnklab-a bnklab-b)

# Pre-clean leftovers from any earlier aborted run.
pkill -f "$WORK/bnk-server serve" 2>/dev/null || true
pkill -f "$WORK/bnk run" 2>/dev/null || true
for ns in "${NS[@]}"; do ip netns del "$ns" 2>/dev/null || true; done
ip link del bnklab-br 2>/dev/null || true
rm -rf "$WORK"
mkdir -p "$WORK"/{srv,a,b}
echo "== building binaries"
go build -o "$WORK/bnk-server" ./cmd/bnk-server
go build -o "$WORK/bnk" ./cmd/bnk

cleanup() {
    set +e
    pkill -f "$WORK/bnk-server serve" 2>/dev/null
    pkill -f "$WORK/bnk run" 2>/dev/null
    for ns in "${NS[@]}"; do ip netns del "$ns" 2>/dev/null; done
    ip link del bnklab-br 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "== building topology"
ip link add bnklab-br type bridge
ip addr add 10.99.0.254/24 dev bnklab-br
ip link set bnklab-br up

i=1
for ns in "${NS[@]}"; do
    ip netns add "$ns"
    ip link add "veth-$ns" type veth peer name eth0 netns "$ns"
    ip link set "veth-$ns" master bnklab-br up
    ip netns exec "$ns" ip addr add "10.99.0.$i/24" dev eth0
    ip netns exec "$ns" ip link set eth0 up
    ip netns exec "$ns" ip link set lo up
    i=$((i+1))
done

echo "== starting bnk-server"
ip netns exec bnklab-srv "$WORK/bnk-server" serve --state-dir "$WORK/srv" --listen :8443 &
sleep 1

KEY=$(ip netns exec bnklab-srv "$WORK/bnk-server" key new --reusable --ttl 1h --state-dir "$WORK/srv")
echo "== enrollment key: $KEY"

echo "== starting clients"
ip netns exec bnklab-a "$WORK/bnk" run --server https://10.99.0.1:8443 --key "$KEY" \
    --state-dir "$WORK/a" --socket "$WORK/a/bnk.sock" --name alpha &
ip netns exec bnklab-b "$WORK/bnk" run --server https://10.99.0.1:8443 --key "$KEY" \
    --state-dir "$WORK/b" --socket "$WORK/b/bnk.sock" --name beta &

echo "== waiting for tunnel"
for i in $(seq 1 30); do
    A_IP=$(python3 -c "import json,sys; print(json.load(open('$WORK/a/client.json'))['ip'])" 2>/dev/null || true)
    B_IP=$(python3 -c "import json,sys; print(json.load(open('$WORK/b/client.json'))['ip'])" 2>/dev/null || true)
    [[ -n "${A_IP:-}" && -n "${B_IP:-}" ]] && break
    sleep 1
done
[[ -n "$A_IP" && -n "$B_IP" ]] || { echo "smoke: clients never enrolled" >&2; exit 1; }
echo "== alpha=$A_IP beta=$B_IP"

for i in $(seq 1 30); do
    if ip netns exec bnklab-a ping -c1 -W1 "$B_IP" >/dev/null 2>&1; then
        echo "== PASS: alpha pinged beta over the tunnel"
        ip netns exec bnklab-srv "$WORK/bnk-server" node ls --state-dir "$WORK/srv"
        exit 0
    fi
    sleep 1
done
echo "smoke: FAIL - tunnel never carried a ping" >&2
exit 1
