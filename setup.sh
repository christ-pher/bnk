#!/usr/bin/env bash
# One-shot setup for bnk: checks the machine, installs what is missing, builds the engine, the Python environment
# and the dashboard, and optionally the MTP draft layer.
#
#   ./setup.sh                          interactive
#   ./setup.sh --yes --mtp --models /data/models
#
# Options:
#   --yes            do not ask before installing system packages
#   --no-system      never touch system packages (only check and report)
#   --mtp            build the MTP draft layer (downloads ~5 GB of the official Qwen checkpoint)
#   --mtp-out PATH   where to write it (default /opt/models/bnk/mtp/mtp-q2_0.gguf)
#   --models DIR     where your model GGUFs live (saved to bnk.env for run.sh)
#   --cuda DIR       the CUDA toolkit to build with (default: detected; CUDA 12.x is required for sm_70)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
YES=0; SYSTEM=1; MTP=0; MTP_OUT="/opt/models/bnk/mtp/mtp-q2_0.gguf"; MODELS=""; CUDA_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) YES=1; shift ;;
    --no-system) SYSTEM=0; shift ;;
    --mtp) MTP=1; shift ;;
    --mtp-out) MTP_OUT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --cuda) CUDA_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 1 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; exit 1; }
ask()  { [[ $YES == 1 ]] && return 0; read -r -p "  $1 [Y/n] " a; [[ -z "$a" || "$a" =~ ^[Yy] ]]; }
ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }   # ver_ge HAVE NEED

# ------------------------------------------------------------------------------------------------ 1. machine
bold "1/6  Checking the machine"
[[ "$(uname -s)" == Linux ]] || fail "bnk runs on Linux"
ok "Linux $(uname -r)"
if command -v nvidia-smi >/dev/null; then
  GPU="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | head -1)"
  ok "GPU: $GPU"
  CC="$(echo "$GPU" | cut -d, -f2 | tr -d ' ')"
  [[ "$CC" == "7.0" ]] || warn "bnk is built and tuned for sm_70 (V100); this GPU is sm_${CC/./} - expect to tune it"
else
  fail "no NVIDIA driver found (nvidia-smi); install the driver first"
fi
RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
if (( RAM_GB < 64 )); then warn "${RAM_GB} GB of RAM: every expert is page-locked in RAM (47 GB for IQ3_S, 61 GB for Orca IQ4_XS)"
else ok "${RAM_GB} GB of RAM"; fi
grep -q avx2 /proc/cpuinfo && ok "CPU has AVX2 ($(nproc) threads)" || fail "the CPU expert kernels need AVX2"

# ------------------------------------------------------------------------------------------------ 2. system packages
bold "2/6  System packages"
need=()
command -v g++ >/dev/null || need+=(build-essential)
command -v git >/dev/null || need+=(git)
command -v curl >/dev/null || need+=(curl)
command -v python3 >/dev/null || need+=(python3)
python3 -c "import venv, ensurepip" 2>/dev/null || need+=(python3-venv)
if [[ ${#need[@]} -gt 0 ]]; then
  if [[ $SYSTEM == 1 ]] && command -v apt-get >/dev/null && ask "install ${need[*]} with apt (needs sudo)?"; then
    sudo apt-get update -qq && sudo apt-get install -y -qq "${need[@]}"
    ok "installed ${need[*]}"
  else
    fail "missing: ${need[*]} - install them and run this again"
  fi
fi
GXX="$(g++ -dumpfullversion)"; ver_ge "$GXX" 12 || fail "g++ $GXX is too old (C++20 needs g++ 12 or newer)"
ok "g++ $GXX, $(python3 --version), git $(git --version | cut -d' ' -f3)"
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ver_ge "$PYV" 3.10 || fail "Python $PYV is too old (3.10 or newer)"

# CUDA: 12.x (CUDA 13 dropped sm_70)
if [[ -z "$CUDA_DIR" ]]; then
  for d in /usr/local/cuda-12.8 /usr/local/cuda-12.9 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    [[ -x "$d/bin/nvcc" ]] && { CUDA_DIR="$d"; break; }
  done
  [[ -z "$CUDA_DIR" ]] && command -v nvcc >/dev/null && CUDA_DIR="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ -z "$CUDA_DIR" || ! -x "$CUDA_DIR/bin/nvcc" ]]; then
  echo "  The CUDA 12.x toolkit was not found. On Ubuntu, NVIDIA's repository provides it:"
  echo "    https://developer.nvidia.com/cuda-12-8-0-download-archive  (then: sudo apt install cuda-toolkit-12-8)"
  fail "install CUDA 12.x and run this again (or pass --cuda DIR)"
fi
CUDAV="$("$CUDA_DIR/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
[[ "$CUDAV" == 12.* ]] || fail "CUDA $CUDAV at $CUDA_DIR: bnk needs CUDA 12.x (CUDA 13 cannot target the V100)"
ok "CUDA $CUDAV at $CUDA_DIR"

# ------------------------------------------------------------------------------------------------ 3. python env
bold "3/6  Python environment (.venv)"
[[ -x "$HERE/.venv/bin/python" ]] || python3 -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install -q --upgrade pip
"$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt"
ok "server and tools dependencies installed"
# CMake >= 3.24 and Ninja: the system's if new enough, else from pip
CMAKE=cmake
if ! command -v cmake >/dev/null || ! ver_ge "$(cmake --version | head -1 | awk '{print $3}')" 3.24; then
  "$HERE/.venv/bin/pip" install -q "cmake>=3.24" ninja
  CMAKE="$HERE/.venv/bin/cmake"
fi
command -v ninja >/dev/null || [[ -x "$HERE/.venv/bin/ninja" ]] || "$HERE/.venv/bin/pip" install -q ninja
export PATH="$HERE/.venv/bin:$PATH"
ok "$($CMAKE --version | head -1), ninja $(ninja --version)"

# ------------------------------------------------------------------------------------------------ 4. engine
bold "4/6  Building the engine"
"$CMAKE" -S "$HERE" -B "$HERE/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_DIR/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=70 > "$HERE/build-cmake.log" 2>&1 \
  || { tail -20 "$HERE/build-cmake.log"; fail "cmake configure failed (log: build-cmake.log)"; }
"$CMAKE" --build "$HERE/build" -j "$(nproc)" >> "$HERE/build-cmake.log" 2>&1 \
  || { tail -30 "$HERE/build-cmake.log"; fail "build failed (log: build-cmake.log)"; }
ok "build/bnk"

# ------------------------------------------------------------------------------------------------ 5. dashboard
bold "5/6  Building the dashboard"
NODE_DIR="$(ls -d "$HERE"/.venv/lib/python3*/site-packages/nodejs_wheel 2>/dev/null | head -1)"
if command -v npm >/dev/null; then NPM=(npm)
else NPM=("$NODE_DIR/bin/node" "$NODE_DIR/lib/node_modules/npm/bin/npm-cli.js"); export PATH="$NODE_DIR/bin:$PATH"; fi
(cd "$HERE/serve/web" && "${NPM[@]}" ci --silent && "${NPM[@]}" run build --silent) > /dev/null 2>&1 \
  || warn "the dashboard did not build (the APIs still work; run.sh retries on start)"
[[ -f "$HERE/serve/web/dist/index.html" ]] && ok "serve/web/dist"

# ------------------------------------------------------------------------------------------------ 6. models, MTP
bold "6/6  Models and the MTP draft layer"
ENVF="$HERE/bnk.env"
if [[ -n "$MODELS" ]]; then
  { grep -v '^BNK_MODELS=' "$ENVF" 2>/dev/null || true; echo "BNK_MODELS=\"$MODELS\""; } > "$ENVF.tmp" && mv "$ENVF.tmp" "$ENVF"
  ok "models directory: $MODELS (saved to bnk.env)"
fi
if [[ $MTP == 1 ]]; then
  mkdir -p "$(dirname "$MTP_OUT")"
  "$HERE/.venv/bin/python" "$HERE/tools/build_mtp.py" --out "$MTP_OUT" --cache "$(dirname "$MTP_OUT")/official"
  { grep -v '^BNK_MTP=' "$ENVF" 2>/dev/null || true; echo "BNK_MTP=\"$MTP_OUT\""; } > "$ENVF.tmp" && mv "$ENVF.tmp" "$ENVF"
  ok "MTP draft layer: $MTP_OUT"
elif [[ -f "$MTP_OUT" ]]; then
  ok "MTP draft layer found at $MTP_OUT"
else
  warn "no MTP draft layer yet: ./setup.sh --mtp builds it (speculative decoding is ~25% faster with it)"
fi

echo
bold "Done."
echo "  Start a model:   ./run.sh iq3_s        (or: ./run.sh orca, or ./run.sh /path/to/model-00001-of-0000N.gguf)"
echo "  Dashboard:       http://$(hostname -I 2>/dev/null | awk '{print $1}'):8080"
[[ -z "$MODELS" ]] && echo "  Models:          run.sh looks in \$BNK_MODELS (set it, or: ./setup.sh --models DIR)"
exit 0
