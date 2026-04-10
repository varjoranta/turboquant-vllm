#!/usr/bin/env bash
# PR #5 native-backend cache-spec integration GPU test.
#
# Runs on the GPU instance. Self-contained, idempotent, survives SSH drops
# when launched via nohup.
#
# What it measures (see .plans/native-backend-cache-spec-integration.md):
#
#   Phase 1 — Import sanity
#   Phase 2 — Plugin registration dry run + version capture
#   For each model in $MODEL_LIST:
#     Phase 3 — Start vLLM with --kv-cache-dtype auto
#     Phase 4 — Capture baseline KV cache capacity (N tokens), completion
#     Phase 5 — Start vLLM with --kv-cache-dtype tq3
#     Phase 6 — Capture tq3 KV cache capacity, completion
#     Phase 7 — Timing check: grep that the STR_DTYPE patch ran BEFORE
#               vLLM resolved the dtype / started loading the model
#     Phase 8 — Capacity ratio: assert tq3_tokens / auto_tokens >= 1.8
#
# Result file format (/tmp/tq-test-result.txt): a human summary followed
# by a RESULTS: key=value block that's easy to machine-parse.
#
# Env overrides:
#   MODEL_LIST     comma-separated HF model ids (default two models below)
#   GPU_MEM        gpu_memory_utilization (default 0.85)
#   MAX_MODEL_LEN  max_model_len (default 2048)
#   BLOCK_SIZE     vLLM block size (default 16)
#   WORKDIR        where the code lives (default $HOME/turboquant-vllm)

set -euo pipefail

MODEL_LIST="${MODEL_LIST:-Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-3B}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
WORKDIR="${WORKDIR:-$HOME/turboquant-vllm}"

LOG="/tmp/tq-test.log"
SERVER_LOG="/tmp/tq-server.log"
RESULT="/tmp/tq-test-result.txt"
PIDFILE="/tmp/tq-server.pid"
PORT=8765

# Idempotent: wipe previous state so a re-run starts clean
rm -f "$LOG" "$SERVER_LOG" "$RESULT" "$PIDFILE"
# Kill any stale vLLM process from a previous run
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 1

exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Iseconds) TQ native backend PR #5 GPU test ==="
echo "MODEL_LIST=$MODEL_LIST"
echo "GPU_MEM=$GPU_MEM  MAX_MODEL_LEN=$MAX_MODEL_LEN  BLOCK_SIZE=$BLOCK_SIZE"
echo "WORKDIR=$WORKDIR"
echo

cd "$WORKDIR"
if [[ -f "$HOME/verda-model-bench/.venv/bin/activate" ]]; then
    # verda-setup.sh puts the venv here
    source "$HOME/verda-model-bench/.venv/bin/activate"
    echo "Activated venv: $HOME/verda-model-bench/.venv"
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
    echo "Activated venv: $WORKDIR/.venv"
fi

# Write result file headers and an initial FAIL marker so if we die
# mid-run the polling ssh still gets SOMETHING useful.
{
    echo "=== TQ native backend PR #5 GPU test ==="
    echo "Started: $(date -Iseconds)"
    echo "Host: $(hostname)"
    echo "Models: $MODEL_LIST"
    echo
    echo "STATUS=IN_PROGRESS"
} > "$RESULT"

# ---------------------------------------------------------------------------
# Helper: cleanup trap so the vLLM server doesn't outlive us
# ---------------------------------------------------------------------------
cleanup() {
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[cleanup] killing vllm PID=$pid"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helper: start vllm, wait for /health, write PID to $PIDFILE
# Args: $1 = kv_cache_dtype, $2 = model
# Returns 0 on success, 1 on failure
# ---------------------------------------------------------------------------
start_vllm() {
    local kv_dtype=$1
    local model=$2
    local start_ts end_ts elapsed
    start_ts=$(date +%s)

    # Truncate server log so each start has its own log
    : > "$SERVER_LOG"

    echo "  Launching vllm: model=$model kv_cache_dtype=$kv_dtype"
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --kv-cache-dtype "$kv_dtype" \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-model-len "$MAX_MODEL_LEN" \
        --block-size "$BLOCK_SIZE" \
        --port "$PORT" \
        --trust-remote-code \
        > "$SERVER_LOG" 2>&1 &
    echo $! > "$PIDFILE"
    local pid
    pid=$(cat "$PIDFILE")

    # Poll /health up to 600s
    for i in $(seq 1 300); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  FAIL: server died before becoming healthy (dtype=$kv_dtype)"
            echo "  --- server log tail ---"
            tail -n 60 "$SERVER_LOG"
            return 1
        fi
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            end_ts=$(date +%s)
            elapsed=$((end_ts - start_ts))
            echo "  OK: healthy after ${elapsed}s"
            return 0
        fi
        sleep 2
    done

    echo "  FAIL: /health timeout after 600s (dtype=$kv_dtype)"
    echo "  --- server log tail ---"
    tail -n 100 "$SERVER_LOG"
    return 1
}

# ---------------------------------------------------------------------------
# Helper: stop vllm and wait for port to free
# ---------------------------------------------------------------------------
stop_vllm() {
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            for i in $(seq 1 15); do
                if ! kill -0 "$pid" 2>/dev/null; then break; fi
                sleep 1
            done
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
    fi
    # Belt-and-suspenders in case there are orphans
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2
}

# ---------------------------------------------------------------------------
# Helper: extract KV cache capacity from the current server log
# Returns a number of tokens (estimated from GPU block count × block size)
# Falls back to 0 if no recognizable pattern matches.
# ---------------------------------------------------------------------------
extract_kv_capacity_tokens() {
    # Try several known vLLM log patterns. Dump matches to stderr for debug.
    local n_blocks=""

    # Pattern 1: "# GPU blocks: 12345" (vLLM 0.18-0.19 standard)
    n_blocks=$(grep -oE "# GPU blocks: [0-9]+" "$SERVER_LOG" 2>/dev/null | head -1 | awk '{print $NF}' || true)
    if [[ -n "$n_blocks" ]]; then
        echo "$((n_blocks * BLOCK_SIZE))"
        return 0
    fi

    # Pattern 2: "Usable KV cache memory: X GiB ... N blocks"
    n_blocks=$(grep -oE "num_gpu_blocks[^0-9]*[0-9]+" "$SERVER_LOG" 2>/dev/null | head -1 | grep -oE "[0-9]+$" || true)
    if [[ -n "$n_blocks" ]]; then
        echo "$((n_blocks * BLOCK_SIZE))"
        return 0
    fi

    # Pattern 3: "maximum concurrency for N tokens per request"
    local max_conc
    max_conc=$(grep -oE "Maximum concurrency for [0-9]+ tokens" "$SERVER_LOG" 2>/dev/null | head -1 | grep -oE "[0-9]+" | head -1 || true)
    if [[ -n "$max_conc" ]]; then
        echo "$max_conc"
        return 0
    fi

    # No known pattern matched; return 0 and let caller handle failure
    echo "0"
    return 1
}

# ---------------------------------------------------------------------------
# Helper: send a completion, return 0 if non-empty response
# ---------------------------------------------------------------------------
send_completion() {
    local model=$1
    local out_var=$2
    local resp generated
    resp=$(curl -sf "http://127.0.0.1:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$model\",\"prompt\":\"The capital of France is\",\"max_tokens\":10,\"temperature\":0}" 2>/dev/null || echo "")
    if [[ -z "$resp" ]]; then
        printf -v "$out_var" "%s" "<ERROR: empty response>"
        return 1
    fi
    generated=$(echo "$resp" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["text"])' 2>/dev/null || echo "")
    printf -v "$out_var" "%s" "$generated"
    [[ -n "$generated" ]]
}

# ---------------------------------------------------------------------------
# Phase 1: Import sanity
# ---------------------------------------------------------------------------
echo "[Phase 1/8] Import sanity check"
python3 - <<'PY'
import sys
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    assert hasattr(AttentionBackendEnum, "CUSTOM")
    assert callable(register_backend)
    print("  OK: vLLM registry interface")
except Exception as e:
    print(f"  FAIL: vLLM registry: {e}")
    sys.exit(1)

try:
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    print("  OK: vLLM FullAttentionSpec")
except Exception as e:
    print(f"  FAIL: FullAttentionSpec: {e}")
    sys.exit(1)

try:
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, kv_cache_dtype_str_to_dtype
    print(f"  OK: STR_DTYPE keys before patch: {sorted(STR_DTYPE_TO_TORCH_DTYPE.keys())}")
except Exception as e:
    print(f"  FAIL: STR_DTYPE_TO_TORCH_DTYPE: {e}")
    sys.exit(1)

try:
    import turboquant_vllm
    print(f"  OK: turboquant_vllm at {turboquant_vllm.__file__}")
except Exception as e:
    print(f"  FAIL: turboquant_vllm import: {e}")
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# Phase 2: Plugin dry run + version capture
# ---------------------------------------------------------------------------
echo "[Phase 2/8] Plugin dry run + version capture"
python3 - <<'PY' > /tmp/tq-versions.txt
import sys, json
info = {}
try:
    import torch
    info['torch'] = torch.__version__
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_mem_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
except Exception as e:
    info['torch'] = f'ERR {e}'

try:
    import vllm
    info['vllm'] = vllm.__version__
except Exception as e:
    info['vllm'] = f'ERR {e}'

try:
    import turboquant_vllm
    info['turboquant_vllm'] = turboquant_vllm.__file__
except Exception as e:
    info['turboquant_vllm'] = f'ERR {e}'

try:
    from turboquant_vllm._vllm_plugin import _register_native_backend, _TQ_DTYPE_NAMES
    ok = _register_native_backend()
    info['native_backend_registered'] = bool(ok)
    info['tq_dtype_names'] = list(_TQ_DTYPE_NAMES)
    # Check STR_DTYPE patch landed
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    info['str_dtype_has_tq3'] = 'tq3' in STR_DTYPE_TO_TORCH_DTYPE
except Exception as e:
    info['plugin_error'] = str(e)

info['python'] = sys.version.split()[0]
print(json.dumps(info, indent=2))
PY
cat /tmp/tq-versions.txt

# Extract versions for the result file
VLLM_VER=$(python3 -c "import json; print(json.load(open('/tmp/tq-versions.txt')).get('vllm', '?'))")
TORCH_VER=$(python3 -c "import json; print(json.load(open('/tmp/tq-versions.txt')).get('torch', '?'))")
TQ_PATH=$(python3 -c "import json; print(json.load(open('/tmp/tq-versions.txt')).get('turboquant_vllm', '?'))")
GPU_NAME=$(python3 -c "import json; print(json.load(open('/tmp/tq-versions.txt')).get('gpu', '?'))")
GPU_MEM_GB=$(python3 -c "import json; print(json.load(open('/tmp/tq-versions.txt')).get('gpu_mem_gb', '?'))")
PYTHON_VER=$(python3 -c "import json; print(json.load(open('/tmp/tq-versions.txt')).get('python', '?'))")
STR_DTYPE_PATCHED=$(python3 -c "import json; print(json.load(open('/tmp/tq-versions.txt')).get('str_dtype_has_tq3', False))")

{
    echo "VLLM_VERSION=$VLLM_VER"
    echo "TORCH_VERSION=$TORCH_VER"
    echo "TURBOQUANT_VLLM_PATH=$TQ_PATH"
    echo "GPU_NAME=$GPU_NAME"
    echo "GPU_MEM_GB=$GPU_MEM_GB"
    echo "PYTHON_VERSION=$PYTHON_VER"
    echo "STR_DTYPE_PATCHED=$STR_DTYPE_PATCHED"
} >> "$RESULT"

# Installed turboquant-vllm commit SHA (if it's a git checkout)
if [[ -d "$WORKDIR/.git" ]]; then
    TQ_SHA=$(git -C "$WORKDIR" rev-parse HEAD 2>/dev/null || echo "unknown")
    TQ_BRANCH=$(git -C "$WORKDIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    echo "TURBOQUANT_VLLM_SHA=$TQ_SHA" >> "$RESULT"
    echo "TURBOQUANT_VLLM_BRANCH=$TQ_BRANCH" >> "$RESULT"
fi

# ---------------------------------------------------------------------------
# Per-model loop: Phases 3-8 for each model in MODEL_LIST
# ---------------------------------------------------------------------------
IFS=',' read -ra MODELS_ARR <<< "$MODEL_LIST"
MODEL_IDX=0
ALL_PASS=1
OVERALL_REASON=""

for MODEL in "${MODELS_ARR[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))
    MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')
    echo
    echo "================================================================================"
    echo "Model $MODEL_IDX: $MODEL"
    echo "================================================================================"

    # --------------------------------------------------------------------------
    # Phase 3 + 4: baseline (auto) run
    # --------------------------------------------------------------------------
    echo "[Phase 3/8] Start vllm --kv-cache-dtype auto for $MODEL"
    if ! start_vllm "auto" "$MODEL"; then
        echo "  FAIL: auto baseline vllm start failed"
        cp "$SERVER_LOG" "/tmp/tq-server-${MODEL_SLUG}-auto.log"
        {
            echo "MODEL_${MODEL_IDX}=$MODEL"
            echo "MODEL_${MODEL_IDX}_AUTO_START=FAIL"
        } >> "$RESULT"
        ALL_PASS=0
        OVERALL_REASON="${OVERALL_REASON:+$OVERALL_REASON, }$MODEL auto start failed"
        continue
    fi

    # Capture the "kv cache allocation" log line(s) for debug
    KV_LINES_AUTO=$(grep -iE "# GPU blocks|kv cache|Usable|num_gpu_blocks|maximum concurrency" "$SERVER_LOG" 2>/dev/null | head -5 || true)
    echo "  KV cache lines (auto):"
    echo "$KV_LINES_AUTO" | sed 's/^/    /'

    AUTO_TOKENS=$(extract_kv_capacity_tokens)
    echo "  AUTO capacity (estimated tokens): $AUTO_TOKENS"

    AUTO_GEN=""
    if send_completion "$MODEL" AUTO_GEN; then
        echo "  AUTO generation: '$AUTO_GEN'"
    else
        echo "  WARN: auto completion failed: '$AUTO_GEN'"
    fi

    # Save full server log for this run
    cp "$SERVER_LOG" "/tmp/tq-server-${MODEL_SLUG}-auto.log"
    stop_vllm

    # --------------------------------------------------------------------------
    # Phase 5 + 6: tq3 run
    # --------------------------------------------------------------------------
    echo "[Phase 5/8] Start vllm --kv-cache-dtype tq3 for $MODEL"
    if ! start_vllm "tq3" "$MODEL"; then
        echo "  FAIL: tq3 vllm start failed"
        cp "$SERVER_LOG" "/tmp/tq-server-${MODEL_SLUG}-tq3.log"
        {
            echo "MODEL_${MODEL_IDX}=$MODEL"
            echo "MODEL_${MODEL_IDX}_AUTO_TOKENS=$AUTO_TOKENS"
            echo "MODEL_${MODEL_IDX}_TQ3_START=FAIL"
        } >> "$RESULT"
        ALL_PASS=0
        OVERALL_REASON="${OVERALL_REASON:+$OVERALL_REASON, }$MODEL tq3 start failed"
        continue
    fi

    # --------------------------------------------------------------------------
    # Phase 7: Timing check — was STR_DTYPE patch line BEFORE model load?
    # --------------------------------------------------------------------------
    PATCH_LINE=$(grep -n "patched STR_DTYPE_TO_TORCH_DTYPE" "$SERVER_LOG" 2>/dev/null | head -1 | cut -d: -f1 || true)
    LOAD_LINE=$(grep -nE "Loading (weights|model|safetensors)|model_executor|init model" "$SERVER_LOG" 2>/dev/null | head -1 | cut -d: -f1 || true)
    SPEC_PATCH_LINE=$(grep -n "patched AttentionLayer.get_kv_cache_spec" "$SERVER_LOG" 2>/dev/null | head -1 | cut -d: -f1 || true)
    SIG_WARN=$(grep -c "signature unexpected" "$SERVER_LOG" 2>/dev/null || echo "0")

    echo "[Phase 7/8] Timing check"
    echo "  STR_DTYPE patch log line: ${PATCH_LINE:-<missing>}"
    echo "  Model loading log line:   ${LOAD_LINE:-<missing>}"
    echo "  Spec patch log line:      ${SPEC_PATCH_LINE:-<missing>}"
    echo "  Signature warnings:       $SIG_WARN"

    TIMING_STATUS="UNKNOWN"
    if [[ -z "$PATCH_LINE" ]]; then
        TIMING_STATUS="FAIL_NO_PATCH_LINE"
    elif [[ -n "$LOAD_LINE" && "$PATCH_LINE" -gt "$LOAD_LINE" ]]; then
        TIMING_STATUS="FAIL_LATE"
    elif [[ "$SIG_WARN" -gt 0 ]]; then
        TIMING_STATUS="FAIL_SIG_GUARD"
    else
        TIMING_STATUS="PASS"
    fi
    echo "  Timing: $TIMING_STATUS"

    # Capture kv-cache allocation lines for tq3
    KV_LINES_TQ3=$(grep -iE "# GPU blocks|kv cache|Usable|num_gpu_blocks|maximum concurrency" "$SERVER_LOG" 2>/dev/null | head -5 || true)
    echo "  KV cache lines (tq3):"
    echo "$KV_LINES_TQ3" | sed 's/^/    /'

    TQ3_TOKENS=$(extract_kv_capacity_tokens)
    echo "  TQ3 capacity (estimated tokens): $TQ3_TOKENS"

    TQ3_GEN=""
    if send_completion "$MODEL" TQ3_GEN; then
        echo "  TQ3 generation: '$TQ3_GEN'"
    else
        echo "  WARN: tq3 completion failed: '$TQ3_GEN'"
    fi

    cp "$SERVER_LOG" "/tmp/tq-server-${MODEL_SLUG}-tq3.log"
    stop_vllm

    # --------------------------------------------------------------------------
    # Phase 8: Capacity ratio check
    # --------------------------------------------------------------------------
    RATIO_STATUS="UNKNOWN"
    RATIO="0.0"
    if [[ "$AUTO_TOKENS" -gt 0 && "$TQ3_TOKENS" -gt 0 ]]; then
        RATIO=$(python3 -c "print(round($TQ3_TOKENS / $AUTO_TOKENS, 3))")
        RATIO_PASS=$(python3 -c "print(1 if $TQ3_TOKENS / $AUTO_TOKENS >= 1.8 else 0)")
        if [[ "$RATIO_PASS" == "1" ]]; then
            RATIO_STATUS="PASS"
        else
            RATIO_STATUS="FAIL_RATIO_TOO_LOW"
        fi
    else
        RATIO_STATUS="FAIL_EXTRACTION"
    fi

    echo "[Phase 8/8] Capacity ratio: $RATIO ($RATIO_STATUS)"

    # Append per-model results to result file
    {
        echo ""
        echo "# Model $MODEL_IDX"
        echo "MODEL_${MODEL_IDX}=$MODEL"
        echo "MODEL_${MODEL_IDX}_AUTO_TOKENS=$AUTO_TOKENS"
        echo "MODEL_${MODEL_IDX}_TQ3_TOKENS=$TQ3_TOKENS"
        echo "MODEL_${MODEL_IDX}_RATIO=$RATIO"
        echo "MODEL_${MODEL_IDX}_RATIO_STATUS=$RATIO_STATUS"
        echo "MODEL_${MODEL_IDX}_TIMING_STATUS=$TIMING_STATUS"
        echo "MODEL_${MODEL_IDX}_SIG_WARN_COUNT=$SIG_WARN"
        echo "MODEL_${MODEL_IDX}_AUTO_GEN=$(echo "$AUTO_GEN" | head -c 200 | tr '\n' ' ')"
        echo "MODEL_${MODEL_IDX}_TQ3_GEN=$(echo "$TQ3_GEN" | head -c 200 | tr '\n' ' ')"
    } >> "$RESULT"

    if [[ "$RATIO_STATUS" != "PASS" || "$TIMING_STATUS" != "PASS" ]]; then
        ALL_PASS=0
        OVERALL_REASON="${OVERALL_REASON:+$OVERALL_REASON, }$MODEL ratio=$RATIO timing=$TIMING_STATUS"
    fi
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo
echo "================================================================================"
if [[ "$ALL_PASS" == "1" ]]; then
    echo "=== ALL PHASES PASS ==="
    FINAL_STATUS="PASS"
else
    echo "=== FAIL: $OVERALL_REASON ==="
    FINAL_STATUS="FAIL"
fi
echo "================================================================================"

{
    echo ""
    echo "=== FINAL ==="
    echo "STATUS=$FINAL_STATUS"
    echo "REASON=${OVERALL_REASON:-all models passed}"
    echo "Finished: $(date -Iseconds)"
} >> "$RESULT"

cat "$RESULT"

# Exit 0 so the nohup wrapper finishes cleanly even on test failure;
# the test verdict lives in $RESULT.
exit 0
