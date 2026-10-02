#!/usr/bin/env bash
# Start the bnk server.
#   ./run.sh [iq3_s|orca|PATH-to-first-shard.gguf] [--port 8080] [--ctx 65536] [--no-mtp] [server options...]
#   BNK_DRAFT_VOCAB= (empty) drafts over the whole vocabulary, e.g. for non-English chats
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODELS="${BNK_MODELS:-/opt/models/Strata/models}"
MTP_DEFAULT="${BNK_MTP:-/opt/models/Strata/runtime/mtp/mtp-q2_0.gguf}"
# the drafter's token subset (English and code; drafts outside it are never proposed, never wrong). "" = all tokens
DRAFT_VOCAB="${BNK_DRAFT_VOCAB-/opt/engines/Strata/data/draft_vocab.bin}"
PROFILE_DEFAULT="${BNK_PROFILE:-/opt/engines/Strata/data/expert-profile.bin}"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/bnk"
mkdir -p "$CACHE"

choice="${1:-iq3_s}"
[[ $# -gt 0 ]] && shift
case "$choice" in
  iq3_s|IQ3_S) MODEL="$MODELS/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"; NAME=iq3_s ;;
  orca|orca-iq4_xs|iq4_xs) MODEL="$MODELS/orca-iq4_xs/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"; NAME=orca-iq4_xs ;;
  *.gguf) MODEL="$choice"; NAME="$(basename "$choice" .gguf)" ;;
  *) echo "unknown model '$choice' (iq3_s, orca, or a .gguf path)"; exit 1 ;;
esac

PORT=8080; CTX=65536; MTP="$MTP_DEFAULT"; EXTRA=()
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
  "$HERE/.venv/bin/pip" install -q regex jinja2 numpy
fi

COUNTS="$CACHE/counts-$NAME.bnkc"
ARGS=(--model "$MODEL" --ctx "$CTX" --port "$PORT" --counts "$COUNTS" --log "$CACHE/engine-$NAME.log")
[[ -n "$MTP" && -f "$MTP" ]] && ARGS+=(--mtp "$MTP")
[[ -n "$MTP" && -n "$DRAFT_VOCAB" && -f "$DRAFT_VOCAB" ]] && ARGS+=(--draft-vocab "$DRAFT_VOCAB")
# the learned routing counts rank the initial VRAM cache once they exist; until then Strata's profile does
[[ ! -f "$COUNTS" && -f "$PROFILE_DEFAULT" ]] && ARGS+=(--profile "$PROFILE_DEFAULT")
cd "$HERE"
exec "$HERE/.venv/bin/python" -m serve.server "${ARGS[@]}" "${EXTRA[@]}"
