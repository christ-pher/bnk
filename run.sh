#!/usr/bin/env bash
# Start the bnk server.
#   ./run.sh                     # no model arg on a terminal: pick from a dropdown of the available models
#   ./run.sh [iq3_s|orca|cyber-frost|abliterated|swift|swift-abliterated|<full-model-id>|PATH-to-first-shard.gguf] [--port 8080] [--ctx 262144] [--no-mtp] [server options...]
#   BNK_DRAFT_VOCAB= (empty) drafts over the whole vocabulary, e.g. for non-English chats
#   BNK_THINK_GUARD=0  turn off the thinking-loop guard (on by default; serve/loop_guard.py)
#   BNK_LOG_LEVEL=quiet|info|debug  terminal output (default info: a line per request, live status while generating)
#   BNK_SLOTS=3        conversations decoded at once, batched (several agents); 1 = one at a time
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

# Interactive model picker (run.sh with no model argument, on a terminal): a dropdown over the available configs,
# each shown by its full model id. whiptail gives an arrow-key box; otherwise fall back to the bash `select` menu.
pick_model() {
  local -a ids
  mapfile -t ids < <(for f in "$HERE/configs"/*.json; do [[ -e "$f" ]] && basename "$f" .json; done | sort)
  [[ ${#ids[@]} -eq 0 ]] && return 1
  if command -v whiptail >/dev/null 2>&1; then
    local -a menu=(); local id
    for id in "${ids[@]}"; do menu+=("$id" ""); done
    whiptail --title "bnk — select a model" \
      --menu "Choose a model to serve (Esc to cancel):" 18 90 "${#ids[@]}" "${menu[@]}" 3>&1 1>&2 2>&3
  else
    local id PS3="Select a model by number (Ctrl-C to cancel): "
    select id in "${ids[@]}"; do [[ -n "$id" ]] && { printf '%s\n' "$id"; return 0; }; done
    return 1
  fi
}

if [[ $# -gt 0 ]]; then
  choice="$1"; shift
elif [[ -t 1 ]]; then
  choice="$(pick_model)" || { echo "no model selected"; exit 1; }
else
  choice="iq3_s"   # non-interactive with no argument: the default
fi
PLE="${BNK_PLE_GGUF:-}"
case "$choice" in
  iq3_s|IQ3_S|Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S) MODEL="$MODELS/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"; NAME=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S ;;
  orca|orca-iq4_xs|iq4_xs|Qwen3.8-Flash-Next-Uncensored-IQ4_XS) MODEL="$MODELS/orca-iq4_xs/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"; NAME=Qwen3.8-Flash-Next-Uncensored-IQ4_XS ;;
  # SC117's GSQ-RCO IQ3_S with Orca's abliterated tensors transplanted (huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)
  abliterated|iq3_s-abliterated|sc117|Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S)
    MODEL="${BNK_ABLITERATED:-/opt/models/gsq-rco-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf}"; NAME=Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S ;;
  # ukisai Swift 1.5 GSQ-RCO IQ3_XXS (huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), highest tier offered
  swift|swift-iq3_xxs|iq3_xxs|Swift-1.5-GSQ-RCO-IQ3_XXS)
    MODEL="${BNK_SWIFT:-/opt/models/swift-gsq-rco/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}"; NAME=Swift-1.5-GSQ-RCO-IQ3_XXS ;;
  # SC117's Swift 1.5 IQ3_XXS with Orca's abliterated tensors transplanted (huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)
  swift-abliterated|swift-abl|Swift-1.5-GSQ-RCO-abliterated-IQ3_XXS)
    MODEL="${BNK_SWIFT_ABLITERATED:-/opt/models/swift-gsq-rco-abliterated/IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf}"; NAME=Swift-1.5-GSQ-RCO-abliterated-IQ3_XXS ;;
  # CYBER-FROST-3.8 PS-GUFF (Q5_K_M), fetched without its PLE table (tools/fetch_gguf.py, see README): that table
  # is byte-identical to Orca's, so it is read from the Orca file
  cyber-frost|cyberfrost|frost|CYBER-FROST-3.8-PS-Q5_K_M)
    MODEL="${BNK_CYBER_FROST:-/opt/models/cyber-frost/CYBER-FROST-3.8-PS-Q5_K_M-noPLE.gguf}"; NAME=CYBER-FROST-3.8-PS-Q5_K_M
    PLE="${BNK_PLE_GGUF:-$MODELS/orca-iq4_xs/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf}" ;;
  *.gguf) MODEL="$choice"; NAME="$(basename "$choice" .gguf)" ;;
  *) echo "unknown model '$choice' (iq3_s, orca, cyber-frost, abliterated, swift, swift-abliterated, or a .gguf path)"; exit 1 ;;
esac

# --port, --ctx and every other server option pass through (defaults: 8080, 262144); a model's persistent server
# options live in configs/<name>.json under "server_args"
MTP="$MTP_DEFAULT"; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
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
ARGS=(--model "$MODEL" --counts "$COUNTS" --log "$CACHE/engine-$NAME.log")
# per-model tuning: sampling defaults, speculation, the thinking-loop guard (configs/<name>.json)
[[ -f "$HERE/configs/$NAME.json" ]] && ARGS+=(--config "$HERE/configs/$NAME.json")
[[ -n "$MTP" ]] && ARGS+=(--mtp "$MTP")
[[ -n "$PLE" ]] && ARGS+=(--ple-gguf "$PLE")
[[ -n "$MTP" && -n "$DRAFT_VOCAB" && -f "$DRAFT_VOCAB" ]] && ARGS+=(--draft-vocab "$DRAFT_VOCAB")
# the learned routing counts (written as the server runs) rank the initial VRAM cache; on a model's first run the
# cache starts unranked and adapts to the traffic
cd "$HERE"
exec "$HERE/.venv/bin/python" -m serve.server "${ARGS[@]}" "${EXTRA[@]}"
