#!/usr/bin/env bash
# Local launcher: rsync the PR #5 branch to a Verda instance, install it
# over the main version that verda-setup.sh already put there, and launch
# tests/gpu/test_native_backend_gpu.sh via nohup + disown.
#
# Non-negotiables (learned the hard way):
#   - No heredocs in ssh commands — all multi-line content lives in files
#     that get rsynced
#   - Always nohup + disown on the remote
#   - Always a pre-flight syntax + import check before the long-running test
#
# Prereq: verda-setup.sh from verda-model-bench has already installed vllm,
# transformers, and turboquant-vllm-from-main into $HOME/verda-model-bench/.venv
# on the target instance.
#
# Usage:
#   ./tests/gpu/run_on_verda.sh <ip>
#   MODEL_LIST=Qwen/Qwen2.5-0.5B ./tests/gpu/run_on_verda.sh <ip>
#
# After launch, poll with:
#   ssh -i ~/.ssh/id_ed25519_varjosoft_hez root@<ip> 'cat /tmp/tq-test-result.txt'

set -euo pipefail

IP="${1:?Usage: $0 <verda-ip>}"
MODEL_LIST_ENV="${MODEL_LIST:-Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-3B}"
GPU_MEM_ENV="${GPU_MEM:-0.85}"
MAX_MODEL_LEN_ENV="${MAX_MODEL_LEN:-2048}"

SSH_KEY="$HOME/.ssh/id_ed25519_varjosoft_hez"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30)

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_DIR='$HOME/turboquant-vllm'

echo "=== PR #5 GPU test launcher ==="
echo "instance:    $IP"
echo "models:      $MODEL_LIST_ENV"
echo "gpu_mem:     $GPU_MEM_ENV"
echo "max_len:     $MAX_MODEL_LEN_ENV"
echo

# Pre-flight: can we reach the instance?
if ! ssh "${SSH_OPTS[@]}" root@"$IP" 'echo reachable' >/dev/null 2>&1; then
    echo "ERROR: cannot ssh root@$IP"
    exit 1
fi

# ----------------------------------------------------------------------
# Step 1: rsync the PR #5 source tree to the remote
# ----------------------------------------------------------------------
echo "[1/5] rsync source tree to ~/turboquant-vllm/"
rsync -az --delete \
    --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    --exclude='build' --exclude='dist' --exclude='*.egg-info' \
    --exclude='.pytest_cache' --exclude='.ruff_cache' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$HERE/" "root@$IP:$HOME/turboquant-vllm/"

# ----------------------------------------------------------------------
# Step 2: generate the env file locally and rsync it to /tmp/tq-test.env
# ----------------------------------------------------------------------
echo "[2/5] rsync env file"
TMP_ENV=$(mktemp -t tq-test-env.XXXXXX)
trap 'rm -f "$TMP_ENV"' EXIT
cat > "$TMP_ENV" <<EOF
export MODEL_LIST='$MODEL_LIST_ENV'
export GPU_MEM='$GPU_MEM_ENV'
export MAX_MODEL_LEN='$MAX_MODEL_LEN_ENV'
export WORKDIR='$HOME/turboquant-vllm'
EOF
rsync -az -e "ssh ${SSH_OPTS[*]}" "$TMP_ENV" "root@$IP:/tmp/tq-test.env"

# ----------------------------------------------------------------------
# Step 3: install the rsync'd tree OVER the main version that verda-setup
#         already installed. --no-deps so we don't touch torch / vllm /
#         transformers. --force-reinstall so we override the existing
#         turboquant-plus-vllm package with this branch's code.
# ----------------------------------------------------------------------
echo "[3/5] install PR #5 branch over main"
ssh "${SSH_OPTS[@]}" root@"$IP" "bash -l -c 'source \$HOME/verda-model-bench/.venv/bin/activate && uv pip install -e \$HOME/turboquant-vllm/ --no-deps --force-reinstall'"

# ----------------------------------------------------------------------
# Step 4: pre-flight — bash syntax check + turboquant_vllm import check
# ----------------------------------------------------------------------
echo "[4/5] pre-flight checks"
ssh "${SSH_OPTS[@]}" root@"$IP" 'bash -n $HOME/turboquant-vllm/tests/gpu/test_native_backend_gpu.sh' \
    || { echo "FAIL: remote test script has syntax errors"; exit 1; }

# Verify the installed turboquant_vllm resolves to the rsync'd path AND the
# cache-spec patch functions from PR #5 are present. The preflight check is
# a rsynced file (no multi-line scripts over ssh).
ssh "${SSH_OPTS[@]}" root@"$IP" 'chmod +x $HOME/turboquant-vllm/tests/gpu/_preflight.sh && $HOME/turboquant-vllm/tests/gpu/_preflight.sh' \
    || { echo "FAIL: preflight check (PR #5 code not installed correctly)"; exit 1; }

# ----------------------------------------------------------------------
# Step 5: launch the test via nohup + disown, no heredocs
# ----------------------------------------------------------------------
echo "[5/5] launch test in background (nohup + disown)"
ssh "${SSH_OPTS[@]}" root@"$IP" 'chmod +x $HOME/turboquant-vllm/tests/gpu/test_native_backend_gpu.sh'

# The remote command is a single line that:
#   1. sources the venv activation (verda-setup.sh's venv)
#   2. sources /tmp/tq-test.env for MODEL_LIST etc
#   3. execs test_native_backend_gpu.sh
# We run it via nohup + disown so it survives ssh disconnection.
# The outer bash -c is needed because ssh's remote shell won't have the
# venv activated by default, and `source` requires a shell.
REMOTE_CMD='nohup bash -c "source \$HOME/verda-model-bench/.venv/bin/activate && source /tmp/tq-test.env && \$HOME/turboquant-vllm/tests/gpu/test_native_backend_gpu.sh" > /tmp/tq-launcher.log 2>&1 < /dev/null & echo \$! > /tmp/tq-launcher.pid && disown'

ssh "${SSH_OPTS[@]}" root@"$IP" "$REMOTE_CMD"
sleep 1
LAUNCHER_PID=$(ssh "${SSH_OPTS[@]}" root@"$IP" 'cat /tmp/tq-launcher.pid 2>/dev/null || echo ?')
echo "launcher PID=$LAUNCHER_PID"

echo
echo "=== launched ==="
echo "Poll test log:   ssh -i $SSH_KEY root@$IP 'tail -n50 /tmp/tq-test.log'"
echo "Poll server log: ssh -i $SSH_KEY root@$IP 'tail -n80 /tmp/tq-server.log'"
echo "Final result:    ssh -i $SSH_KEY root@$IP 'cat /tmp/tq-test-result.txt'"
echo "Is running:      ssh -i $SSH_KEY root@$IP 'pgrep -af test_native_backend_gpu || echo NOT_RUNNING'"
echo
echo "Expected wall time: ~8-10 min (2 models × 2 vllm starts)"
echo "Don't forget to 'discontinue' the instance when done (not 'stop')."
