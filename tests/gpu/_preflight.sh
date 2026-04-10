#!/usr/bin/env bash
# Remote preflight check. Rsynced + executed by run_on_verda.sh.
# Verifies the PR #5 branch is actually installed (not main), and that
# the cache-spec patch functions are present. Exits non-zero on failure.
set -euo pipefail

source "$HOME/verda-model-bench/.venv/bin/activate"

python3 - <<'PY'
import sys
try:
    import turboquant_vllm
except Exception as e:
    print(f"FAIL: cannot import turboquant_vllm: {e}")
    sys.exit(1)

path = turboquant_vllm.__file__
print(f"turboquant_vllm.__file__ = {path}")
if "turboquant-vllm" not in path:
    print(f"FAIL: wrong install path: {path}")
    sys.exit(1)

try:
    from turboquant_vllm._vllm_plugin import (
        _patch_get_kv_cache_spec,
        _eager_patch_str_dtype_mapping,
        _patch_mla_fail_loud,
        _TQ_DTYPE_NAMES,
    )
except ImportError as e:
    print(f"FAIL: PR #5 code not active (missing cache-spec patches): {e}")
    sys.exit(1)

print(f"OK: PR #5 patch functions present. Dtypes: {list(_TQ_DTYPE_NAMES)}")
PY
