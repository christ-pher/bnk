#!/usr/bin/env bash
# Start the bnk server.
#   ./run.sh [iq3_s|orca|cyber-frost|PATH-to-first-shard.gguf] [--port 8080] [--ctx 262144] [--no-mtp] [server options...]
#   BNK_DRAFT_VOCAB= (empty) drafts over the whole vocabulary, e.g. for non-English chats
#   BNK_THINK_GUARD=0  turn off the thinking-loop guard (on by default; serve/loop_guard.py)
#   BNK_LOG_LEVEL=quiet|info|debug  terminal output (default info: a line per request, live status while generating)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# settings saved by setup.sh (BNK_MODELS, BNK_MTP); the environment still wins
if [[ -f "$HERE/bnk.env" ]]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^BNK_[A-Z_]+$ && -z "${!k:-}" ]] && export "$k=${v//\"/}"
  done < "$HERE/bnk.env"
fi
MODELS="${BNK_MODELS:-/opt/models/Strata/models}"
# bnk's own MTP draft layer, built from the official Qwen checkpoint by tools/build_mtp.py
MTP_DEFAULT="${BNK_MTP:-/opt/models/bnk/mtp/mtp-q2_0.gguf}"
# the drafter's token subset (tools/draft_vocab.py: Latin scripts, code, symbols, emoji); a token outside it is never
# drafted, never wrong. Set BNK_DRAFT_VOCAB= (empty) to draft over the whole vocabulary.
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/bnk"
mkdir -p "$CACHE"

choice="${1:-iq3_s}"
PLE="${BNK_PLE_GGUF:-}"
[[ $# -gt 0 ]] && shift
case "$choice" in
  iq3_s|IQ3_S) MODEL="$MODELS/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"; NAME=iq3_s ;;
  orca|orca-iq4_xs|iq4_xs) MODEL="$MODELS/orca-iq4_xs/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"; NAME=orca-iq4_xs ;;
  # CYBER-FROST-3.8 PS-GUFF (Q5_K_M), fetched without its PLE table (tools/fetch_gguf.py, see README): that table
  # is byte-identical to Orca's, so it is read from the Orca file
  cyber-frost|cyberfrost|frost)
    MODEL="${BNK_CYBER_FROST:-/opt/models/cyber-frost/CYBER-FROST-3.8-PS-Q5_K_M-noPLE.gguf}"; NAME=cyber-frost
    PLE="${BNK_PLE_GGUF:-$MODELS/orca-iq4_xs/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf}" ;;
  *.gguf) MODEL="$choice"; NAME="$(basename "$choice" .gguf)" ;;
  *) echo "unknown model '$choice' (iq3_s, orca, cyber-frost, or a .gguf path)"; exit 1 ;;
esac

PORT=8080; CTX=262144; MTP="$MTP_DEFAULT"; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --no-mtp) MTP=""; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ ! -x "$HERE/build/bnk" ]]; then
  echo "building bnk ..."
  cmake -S "$HERE" -B "$HERE/build" -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$HERE/build" --target bnk -j "$(nproc)"
fi
if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt"
fi

# the dashboard (serve/web): built once, and again whenever its sources change
WEB="$HERE/serve/web"
if [[ ! -f "$WEB/dist/index.html" || -n "$(find "$WEB/src" "$WEB/index.html" -newer "$WEB/dist/index.html" -print -quit)" ]]; then
  NODE_BIN="$(ls -d "$HERE"/.venv/lib/python3*/site-packages/nodejs_wheel 2>/dev/null | head -1)"
  if command -v npm >/dev/null; then NPM=(npm)
  elif [[ -n "$NODE_BIN" ]]; then NPM=("$NODE_BIN/bin/node" "$NODE_BIN/lib/node_modules/npm/bin/npm-cli.js"); export PATH="$NODE_BIN/bin:$PATH"
  else "$HERE/.venv/bin/pip" install -q nodejs-wheel; NODE_BIN="$(ls -d "$HERE"/.venv/lib/python3*/site-packages/nodejs_wheel | head -1)"
       NPM=("$NODE_BIN/bin/node" "$NODE_BIN/lib/node_modules/npm/bin/npm-cli.js"); export PATH="$NODE_BIN/bin:$PATH"; fi
  echo "building the dashboard ..."
  (cd "$WEB" && { [[ -d node_modules ]] || "${NPM[@]}" ci --silent; } && "${NPM[@]}" run build --silent) >/dev/null || echo "warning: dashboard build failed (the APIs still work)"
fi

DRAFT_VOCAB="${BNK_DRAFT_VOCAB-$CACHE/draft-vocab-$NAME.bin}"
if [[ -n "$MTP" && ! -f "$MTP" ]]; then
  echo "note: no MTP draft layer at $MTP; running without speculative decoding."
  echo "      build it once with: .venv/bin/python tools/build_mtp.py --out $MTP"
  MTP=""
fi
if [[ -n "$MTP" && -n "$DRAFT_VOCAB" && ! -f "$DRAFT_VOCAB" ]]; then
  "$HERE/.venv/bin/python" "$HERE/tools/draft_vocab.py" --gguf "$MODEL" --out "$DRAFT_VOCAB" >/dev/null
fi

COUNTS="$CACHE/counts-$NAME.bnkc"
ARGS=(--model "$MODEL" --ctx "$CTX" --port "$PORT" --counts "$COUNTS" --log "$CACHE/engine-$NAME.log")
# per-model tuning: sampling defaults, speculation, the thinking-loop guard (configs/<name>.json)
[[ -f "$HERE/configs/$NAME.json" ]] && ARGS+=(--config "$HERE/configs/$NAME.json")
[[ -n "$MTP" ]] && ARGS+=(--mtp "$MTP")
[[ -n "$PLE" ]] && ARGS+=(--ple-gguf "$PLE")
[[ -n "$MTP" && -n "$DRAFT_VOCAB" && -f "$DRAFT_VOCAB" ]] && ARGS+=(--draft-vocab "$DRAFT_VOCAB")
# the learned routing counts (written as the server runs) rank the initial VRAM cache; on a model's first run the
# cache starts unranked and adapts to the traffic
cd "$HERE"
exec "$HERE/.venv/bin/python" -m serve.server "${ARGS[@]}" "${EXTRA[@]}"
