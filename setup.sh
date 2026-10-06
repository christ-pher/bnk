#!/usr/bin/env bash
# One-shot setup for bnk: checks the machine, installs what is missing (on Ubuntu also the NVIDIA driver and the
# CUDA 12.8 toolkit), builds the engine, the Python environment and the dashboard, the MTP draft layer, and
# downloads a model.
#
#   ./setup.sh                                      interactive: asks before each install or download
#   ./setup.sh --yes --mtp --download iq3_s         everything, no questions
#
# Options:
#   --yes             do not ask (install system packages, the driver and CUDA, download what was asked for)
#   --no-system       never touch system packages (only check and report)
#   --mtp             build the MTP draft layer (downloads ~5 GB of the official Qwen checkpoint)
#   --mtp-out PATH    where to write it (default: MODELS/mtp/mtp-q2_0.gguf)
#   --models DIR      where model GGUFs live (default ~/models; saved to bnk.env for run.sh)
#   --download LIST   download models: iq3_s (84 GB, the default model), orca (98 GB, gated), abliterated (84 GB)
#   --no-download     never offer to download a model
#   --cuda DIR        the CUDA toolkit to build with (default: detected; CUDA 12.x is required for sm_70)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
YES=0; SYSTEM=1; MTP=0; MTP_OUT=""; MODELS=""; CUDA_DIR=""; DOWNLOAD=""; NO_DL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) YES=1; shift ;;
    --no-system) SYSTEM=0; shift ;;
    --mtp) MTP=1; shift ;;
    --mtp-out) MTP_OUT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --cuda) CUDA_DIR="$2"; shift 2 ;;
    --download) DOWNLOAD="$2"; shift 2 ;;
    --no-download) NO_DL=1; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 1 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; exit 1; }
ask()  { [[ $YES == 1 ]] && return 0; read -r -p "  $1 [Y/n] " a; [[ -z "$a" || "$a" =~ ^[Yy] ]]; }
ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }   # ver_ge HAVE NEED

# NVIDIA's apt repository (driver and CUDA toolkit) on Ubuntu 22.04 / 24.04
UBU=""
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  [[ "${ID:-}" == ubuntu && "$(uname -m)" == x86_64 ]] && case "${VERSION_ID:-}" in 22.04) UBU=ubuntu2204 ;; 24.04) UBU=ubuntu2404 ;; esac
fi
nvidia_repo() {
  dpkg -s cuda-keyring >/dev/null 2>&1 && return 0
  local deb=/tmp/cuda-keyring_1.1-1_all.deb
  curl -fsSL -o "$deb" "https://developer.download.nvidia.com/compute/cuda/repos/$UBU/x86_64/cuda-keyring_1.1-1_all.deb" \
    && sudo dpkg -i "$deb" >/dev/null && sudo apt-get update -qq
}
secure_boot_note() {
  if command -v mokutil >/dev/null && mokutil --sb-state 2>/dev/null | grep -qi enabled; then
    warn "Secure Boot is on: the driver install asks for a password, and the next boot shows a blue"
    warn "'Perform MOK management' screen - choose 'Enroll MOK', type that password, then reboot. Without that"
    warn "step the driver does not load (nvidia-smi fails)."
  fi
}

# ------------------------------------------------------------------------------------------------ 1. machine
bold "1/6  Checking the machine"
[[ "$(uname -s)" == Linux ]] || fail "bnk runs on Linux"
ok "Linux $(uname -r)"
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  warn "no working NVIDIA driver (nvidia-smi)"
  if [[ -n "$UBU" && $SYSTEM == 1 ]] && ask "install the NVIDIA driver and the CUDA 12.8 toolkit from NVIDIA's repository (needs sudo, then a reboot)?"; then
    secure_boot_note
    sudo apt-get install -y -qq curl ca-certificates >/dev/null
    nvidia_repo || fail "could not add NVIDIA's repository"
    sudo apt-get install -y cuda-drivers cuda-toolkit-12-8 || fail "the driver install failed (see above)"
    echo
    bold "The driver is installed. Reboot now (sudo reboot), then run ./setup.sh again to finish."
    exit 0
  fi
  echo "  Install the NVIDIA driver (and the CUDA 12.8 toolkit), reboot, and run this again. On Ubuntu:"
  echo "    https://developer.nvidia.com/cuda-12-8-0-download-archive  ->  sudo apt install cuda-drivers cuda-toolkit-12-8"
  fail "no NVIDIA driver"
fi
GPU="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | head -1)"
ok "GPU: $GPU"
CC="$(echo "$GPU" | cut -d, -f2 | tr -d ' ')"
[[ "$CC" == "7.0" ]] || warn "bnk is built and tuned for sm_70 (V100); this GPU is sm_${CC/./} - expect to tune it"
DRV="$(echo "$GPU" | cut -d, -f4 | tr -d ' ')"
if ! ver_ge "$DRV" 570; then
  warn "driver $DRV is older than CUDA 12.8 needs (570 or newer): sudo apt install cuda-drivers, then reboot"
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
command -v whiptail >/dev/null || need+=(whiptail)   # run.sh's no-arg model picker
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
if [[ ( -z "$CUDA_DIR" || ! -x "$CUDA_DIR/bin/nvcc" ) && -n "$UBU" && $SYSTEM == 1 ]] && ask "the CUDA 12.8 toolkit is missing: install it from NVIDIA's repository (needs sudo)?"; then
  nvidia_repo && sudo apt-get install -y cuda-toolkit-12-8 && CUDA_DIR=/usr/local/cuda-12.8
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
  || warn "the dashboard did not build (the APIs still work; run ./setup.sh again to retry)"
[[ -f "$HERE/serve/web/dist/index.html" ]] && ok "serve/web/dist"

# ------------------------------------------------------------------------------------------------ 6. models, MTP
bold "6/6  Models and the MTP draft layer"
ENVF="$HERE/bnk.env"
if [[ -z "$MODELS" ]]; then   # the folder saved before, else run.sh's built-in one if it holds models, else ~/models
  MODELS="$(sed -n 's/^BNK_MODELS="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$ENVF" 2>/dev/null | tail -1)"
  RUN_DEFAULT="$(sed -n 's/^MODELS="\${BNK_MODELS:-\(.*\)}"$/\1/p' "$HERE/run.sh" | head -1)"
  [[ -z "$MODELS" && -n "$RUN_DEFAULT" ]] && ls "$RUN_DEFAULT"/*/*.gguf >/dev/null 2>&1 && MODELS="$RUN_DEFAULT"
  [[ -z "$MODELS" ]] && MODELS="$HOME/models"
fi
mkdir -p "$MODELS"
{ grep -v '^BNK_MODELS=' "$ENVF" 2>/dev/null || true; echo "BNK_MODELS=\"$MODELS\""; } > "$ENVF.tmp" && mv "$ENVF.tmp" "$ENVF"
ok "models directory: $MODELS (saved to bnk.env)"
if [[ -z "$DOWNLOAD" && $NO_DL == 0 && ! -f "$MODELS/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" ]]; then
  if [[ $YES == 0 ]] && ask "no model found: download the default model, IQ3_S (84 GB), into $MODELS now?"; then DOWNLOAD=iq3_s; fi
fi
if [[ -n "$DOWNLOAD" ]]; then
  "$HERE/.venv/bin/python" "$HERE/tools/fetch_models.py" ${DOWNLOAD//,/ } --models "$MODELS" \
    || fail "the model download did not finish (run the same command again to resume)"
  ok "models: $DOWNLOAD"
fi
if [[ "$DOWNLOAD" == *abliterated* ]]; then
  { grep -v '^BNK_ABLITERATED=' "$ENVF" 2>/dev/null || true
    echo "BNK_ABLITERATED=\"$MODELS/gsq-rco-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf\""; } > "$ENVF.tmp" && mv "$ENVF.tmp" "$ENVF"
fi
# the draft layer: where it already is (an earlier setup), else inside the models folder
if [[ -z "$MTP_OUT" ]]; then
  MTP_OUT="$(sed -n 's/^BNK_MTP="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$ENVF" 2>/dev/null | tail -1)"
  [[ -z "$MTP_OUT" && -f /opt/models/bnk/mtp/mtp-q2_0.gguf ]] && MTP_OUT=/opt/models/bnk/mtp/mtp-q2_0.gguf
  [[ -z "$MTP_OUT" ]] && MTP_OUT="$MODELS/mtp/mtp-q2_0.gguf"
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
echo "  Models:          $MODELS  (more: .venv/bin/python tools/fetch_models.py --list)"
exit 0
