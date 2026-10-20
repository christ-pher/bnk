#!/usr/bin/env bash
# Start the bnk server.
#   ./run.sh                     # no model arg on a terminal: pick from a dropdown of the models in configs/
#   ./run.sh [NAME|ALIAS|PATH-to-first-shard.gguf] [--port 8080] [--ctx 262144] [--no-mtp] [server options...]
#   Models are configs/<name>.json; ./models.sh lists, adds and downloads them (tools/models.py).
#   BNK_DRAFT_VOCAB= (empty) drafts over the whole vocabulary, e.g. for non-English chats
#   BNK_THINK_GUARD=0  turn off the thinking-loop guard (on by default; serve/loop_guard.py)
#   BNK_LOG_LEVEL=quiet|info|debug  terminal output (default info: a line per request, live status while generating)
#   BNK_SLOTS=3        conversations decoded at once, batched (several agents); 1 = one at a time
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# settings saved by setup.sh and models.sh (BNK_MODELS, BNK_MTP); the environment still wins
if [[ -f "$HERE/bnk.env" ]]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^BNK_[A-Z_]+$ && -z "${!k:-}" ]] && export "$k=${v//\"/}"
  done < "$HERE/bnk.env"
fi
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/bnk"
mkdir -p "$CACHE"
# bnk's own MTP draft layer, built from the official Qwen checkpoint by tools/build_mtp.py
MTP_DEFAULT="${BNK_MTP:-$CACHE/mtp/mtp-q2_0.gguf}"
# the drafter's token subset (tools/draft_vocab.py: Latin scripts, code, symbols, emoji); a token outside it is never
# drafted, never wrong. Set BNK_DRAFT_VOCAB= (empty) to draft over the whole vocabulary.

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt"
fi
MODELS_PY=("$HERE/.venv/bin/python" "$HERE/tools/models.py")

# Interactive model picker (run.sh with no model argument, on a terminal): a dropdown over configs/, each model shown
# by its full name and whether it is downloaded. whiptail gives an arrow-key box; otherwise the bash `select` menu.
pick_model() {
  local -a ids=() menu=(); local id st
  while IFS=$'\t' read -r id st; do ids+=("$id"); menu+=("$id" "$st"); done < <("${MODELS_PY[@]}" menu)
  [[ ${#ids[@]} -eq 0 ]] && { echo "no models in configs/: add one with ./models.sh add REPO" >&2; return 1; }
  if command -v whiptail >/dev/null 2>&1; then
    whiptail --title "bnk — select a model" \
      --menu "Choose a model to serve (Esc to cancel):" 18 100 "${#ids[@]}" "${menu[@]}" 3>&1 1>&2 2>&3
  else
    local PS3="Select a model by number (Ctrl-C to cancel): "
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
PLE="${BNK_PLE_GGUF:-}"   # or the config's "ple_gguf" (a model whose n-gram table is a separate GGUF), read below
if [[ "$choice" == *.gguf ]]; then
  MODEL="$choice"; NAME="$(basename "$choice" .gguf)"
  [[ -f "$MODEL" ]] || { echo "model not found: $MODEL"; exit 1; }
else
  # a config by name or alias: where its files are (BNK_MODELS, else the Hugging Face cache); if they are not
  # downloaded yet, offer to download them on a terminal
  out="$("${MODELS_PY[@]}" resolve "$choice")" && rc=0 || rc=$?
  [[ $rc == 0 || $rc == 2 ]] || exit 1   # 2: configured but not downloaded
  NAME="${out%%$'\t'*}"; MODEL="${out#*$'\t'}"
  if [[ -z "$MODEL" ]]; then
    echo "$NAME is not downloaded yet."
    if [[ -t 0 && -t 1 ]] && read -r -p "download it now? [Y/n] " a && [[ -z "$a" || "$a" =~ ^[Yy] ]]; then
      "${MODELS_PY[@]}" download "$NAME" || exit 1
      out="$("${MODELS_PY[@]}" resolve "$NAME")" || exit 1
      MODEL="${out#*$'\t'}"
    else
      echo "  download it with: ./models.sh download $NAME [--dir DIR]"; exit 1
    fi
  fi
fi

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
  cmake --build "$HERE/build" --target bnk requant_experts -j "$(nproc)"
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

# a model's own drafter: configs/<name>.json "mtp": "" runs without speculation, "builtin" drafts with the MTP block
# inside the model file (the dense qwen35 family), a path names a draft-layer GGUF
if [[ -f "$HERE/configs/$NAME.json" ]]; then
  cfg_mtp=$("$HERE/.venv/bin/python" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["mtp"] if "mtp" in c else "<default>")' "$HERE/configs/$NAME.json")
  [[ "$cfg_mtp" != "<default>" ]] && MTP="$cfg_mtp"
fi
DRAFT_VOCAB="${BNK_DRAFT_VOCAB-$CACHE/draft-vocab-$NAME.bin}"
if [[ -n "$MTP" && "$MTP" != builtin && ! -f "$MTP" ]]; then
  echo "note: no MTP draft layer at $MTP; running without speculative decoding."
  echo "      build it once with: .venv/bin/python tools/build_mtp.py --out $MTP --cache $(dirname "$MTP")/official"
  MTP=""
fi
if [[ -n "$MTP" && -n "$DRAFT_VOCAB" && ! -f "$DRAFT_VOCAB" ]]; then
  "$HERE/.venv/bin/python" "$HERE/tools/draft_vocab.py" --gguf "$MODEL" --out "$DRAFT_VOCAB" >/dev/null
fi

if [[ -z "$PLE" && -f "$HERE/configs/$NAME.json" ]]; then
  PLE=$("$HERE/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ple_gguf",""))' "$HERE/configs/$NAME.json")
  [[ -n "$PLE" && ! -f "$PLE" ]] && { echo "the PLE table $PLE (configs/$NAME.json) is missing" >&2; exit 1; }
fi
# the vision encoder (images in chat requests) when the config names one; BNK_VISION=0 serves text only
MMPROJ=""
if [[ "${BNK_VISION:-1}" != 0 && -f "$HERE/configs/$NAME.json" ]]; then
  MMPROJ="$("${MODELS_PY[@]}" mmproj "$NAME" 2>/dev/null || true)"
  if [[ -z "$MMPROJ" ]] && "$HERE/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])).get("source",{}); sys.exit(0 if s.get("mmproj") or s.get("mmproj_path") or s.get("mmproj_from") else 1)' "$HERE/configs/$NAME.json"; then
    echo "note: the vision encoder is not downloaded; serving text only (./models.sh download $NAME fetches it)"
  fi
fi
COUNTS="$CACHE/counts-$NAME.bnkc"
# cold experts (docs/EXPERT-MIX.md): the config's "cold_experts" names a file in the cache directory. Missing, it is
# written in the background (build/requant_experts, ~16 min on 24 threads) and used from the next start on; present,
# tools/cold_drift.py checks it against today's routing counts and rewrites it in the background when more than a
# tenth of its slots would hold another expert. BNK_COLD_AUTO=0 leaves the file alone either way.
if [[ -f "$HERE/configs/$NAME.json" && "${BNK_COLD_AUTO:-1}" != 0 ]]; then
  COLD=$("$HERE/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cold_experts",""))' "$HERE/configs/$NAME.json")
  if [[ -n "$COLD" ]]; then
    [[ "$COLD" == /* ]] || COLD="$CACHE/$COLD"
    BANDS=$("$HERE/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cold_bands","iq2_s/q2_0:30,iq3_s/q2_0:40"))' "$HERE/configs/$NAME.json")
    regen=""
    if [[ ! -f "$COLD" ]]; then
      [[ -f "$COUNTS" ]] && regen="no cold-experts file yet" || echo "note: no cold-experts file and no routing counts yet; serve once, then restart to write it"
    elif [[ -f "$COUNTS" ]]; then
      drift=$("$HERE/.venv/bin/python" "$HERE/tools/cold_drift.py" "$COLD" "$COUNTS" 2>/dev/null); rc=$?
      [[ -n "$drift" ]] && echo "note: $drift"
      [[ $rc == 3 ]] && regen="the routing drifted"
    fi
    if [[ -n "$regen" && ! -f "$COLD.tmp" ]]; then
      echo "note: $regen: writing $COLD in the background (bands $BANDS); the next start uses it"
      [[ -x "$HERE/build/requant_experts" ]] || cmake --build "$HERE/build" --target requant_experts -j "$(nproc)" >/dev/null
      nohup bash -c "\"$HERE/build/requant_experts\" --model \"$MODEL\" --counts \"$COUNTS\" --out \"$COLD.tmp\" --bands \"$BANDS\" --threads 12 > \"$COLD.log\" 2>&1 && mv -f \"$COLD.tmp\" \"$COLD\" || rm -f \"$COLD.tmp\"" >/dev/null 2>&1 &
    fi
  fi
fi
ARGS=(--model "$MODEL" --counts "$COUNTS" --log "$CACHE/engine-$NAME.log")
# per-model tuning: sampling defaults, speculation, the thinking-loop guard (configs/<name>.json)
[[ -f "$HERE/configs/$NAME.json" ]] && ARGS+=(--config "$HERE/configs/$NAME.json")
[[ -n "$MTP" ]] && ARGS+=(--mtp "$MTP")
[[ -n "$PLE" ]] && ARGS+=(--ple-gguf "$PLE")
[[ -n "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ")
[[ -n "$MTP" && -n "$DRAFT_VOCAB" && -f "$DRAFT_VOCAB" ]] && ARGS+=(--draft-vocab "$DRAFT_VOCAB")
# the learned routing counts (written as the server runs) rank the initial VRAM cache; on a model's first run the
# cache starts unranked and adapts to the traffic
cd "$HERE"
exec "$HERE/.venv/bin/python" -m serve.server "${ARGS[@]}" "${EXTRA[@]}"
