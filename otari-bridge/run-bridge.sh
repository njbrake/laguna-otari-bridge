#!/usr/bin/env bash
# Single-pane launcher for the Laguna-S 2.1 <-> Otari bridge.
#
# Starts, in one tmux pane:
#   1. llama-server with Metal (bound to 127.0.0.1 -- only the proxy reaches it)
#   2. Caddy auth gate on :9000 (bearer-token check from ./.token)
#   3. Tailscale Funnel -> :9000  (public HTTPS URL)
#
# Ctrl-C tears all three down (and turns Funnel off so nothing public lingers).
#
# NOTE: this reuses the same ports (8000/9000) and the same Tailscale Funnel as
# the ds4 bridge. Only one of the two can be exposed at a time -- stop ds4
# before running this.
set -uo pipefail

BRIDGE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$BRIDGE_DIR")"
TOKEN_FILE="$BRIDGE_DIR/.token"
CADDYFILE="$BRIDGE_DIR/Caddyfile"
CADDY_BIN="$BRIDGE_DIR/caddy"
LLAMA_BIN="$REPO_DIR/llama.cpp/build/bin/llama-server"
MODEL="$REPO_DIR/models/laguna-s-2.1-Q4_K_M.gguf"
DRAFT="$REPO_DIR/models/laguna-s-2.1-DFlash-BF16.gguf"
LLAMA_LOG="$BRIDGE_DIR/llama-server.log"
CADDY_LOG="$BRIDGE_DIR/caddy.log"
PROXY_PORT=9000

# Context. Laguna-S is 48 layers, but only 12 are global-attention; the other
# 36 use a 512-token sliding window, so KV cost scales with just those 12.
# 128K costs roughly 7GB of KV -- raise to 262144 for the full 256K if you
# have raised iogpu.wired_limit_mb (see README).
CTX="${CTX:-131072}"

command -v caddy >/dev/null 2>&1 && CADDY_BIN="caddy"

# Tailscale CLI is a zsh alias for you; resolve the real binary for this script.
TS="$(command -v tailscale || true)"
[ -z "$TS" ] && TS="/Applications/Tailscale.app/Contents/MacOS/Tailscale"

# --- preflight ---
[ -f "$TOKEN_FILE" ] || { echo "[bridge] missing $TOKEN_FILE -- run: openssl rand -hex 32 > $TOKEN_FILE"; exit 1; }
[ -x "$LLAMA_BIN" ]  || { echo "[bridge] no llama-server at $LLAMA_BIN -- build llama.cpp first"; exit 1; }
[ -f "$MODEL" ]      || { echo "[bridge] missing model at $MODEL"; exit 1; }
export LLM_API_TOKEN="$(cat "$TOKEN_FILE")"

if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
	echo "[bridge] port 8000 is already in use (ds4-server still running?). Stop it first."; exit 1
fi

FUNNEL_HOST="$("$TS" status --json 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))" 2>/dev/null)"
[ -z "$FUNNEL_HOST" ] && FUNNEL_HOST="<your-node>.<tailnet>.ts.net"

LLAMA_PID=""; CADDY_PID=""
cleanup() {
	echo; echo "[bridge] shutting down..."
	[ -n "$CADDY_PID" ] && kill "$CADDY_PID" 2>/dev/null
	[ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null
	"$TS" funnel --https=443 off 2>/dev/null
	echo "[bridge] funnel off, processes stopped."
}
trap cleanup EXIT INT TERM

# --- 1. llama-server (localhost only, all layers on Metal) ---
# -ngl 999 offloads every layer to the GPU. --jinja is required: Laguna ships a
# Jinja chat template and tool-call parser.
#
# Speculative decoding with the DFlash drafter is DELIBERATELY OFF. The model
# card recommends it, but measured on this M4 Max it is a large net loss:
#
#   spec off            39.5 tok/s
#   spec on, n-max=4    15.9 tok/s
#   spec on, n-max=15    6.9 tok/s
#
# Lower n-max just walks back toward the baseline, so it is overhead rather
# than a tuning problem -- likely the dflash path is not Metal-optimised.
# Set SPEC=1 to re-test it after a fork update.
SPEC_ARGS=()
if [ "${SPEC:-0}" = "1" ]; then
	SPEC_ARGS=(--spec-draft-model "$DRAFT" --spec-type draft-dflash --spec-draft-n-max "${SPEC_N:-4}")
	echo "[bridge] speculative decoding ENABLED (n-max=${SPEC_N:-4}) -- expect it to be slower"
fi

echo "[bridge] starting llama-server (127.0.0.1:8000, ctx=$CTX) -> $LLAMA_LOG"
caffeinate -i "$LLAMA_BIN" \
	--model "$MODEL" \
	--alias laguna-s-2.1 \
	--host 127.0.0.1 --port 8000 \
	--ctx-size "$CTX" \
	--n-gpu-layers 999 \
	--flash-attn on \
	--jinja \
	"${SPEC_ARGS[@]}" \
	> "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!

# --- 2. caddy auth gate ---
echo "[bridge] starting caddy auth gate (:$PROXY_PORT) -> $CADDY_LOG"
"$CADDY_BIN" run --config "$CADDYFILE" > "$CADDY_LOG" 2>&1 &
CADDY_PID=$!

# --- 3. funnel ---
echo "[bridge] ensuring tailscale funnel -> :$PROXY_PORT"
"$TS" funnel --bg "$PROXY_PORT" >/dev/null 2>&1

sleep 1
cat <<EOF

  Public URL : https://$FUNNEL_HOST/v1
  Model id   : openai:laguna-s-2.1   (in Otari)
  Test       : curl https://$FUNNEL_HOST/v1/models -H "Authorization: Bearer \$(cat $TOKEN_FILE)"

[bridge] a 75GB model takes a few minutes to load; expect 502 through the proxy until it is ready.
[bridge] tailing logs -- Ctrl-C stops llama-server + caddy + funnel.
----------------------------------------------------------------
EOF
exec tail -n +1 -f "$LLAMA_LOG" "$CADDY_LOG"
