#!/usr/bin/env bash
# Source from assignment2-systems root:  source scripts/env.sh
# Puts uv/torch/triton caches and the project venv on node-local /tmp (off home quota).

set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_USER="${USER:-$(id -un)}"
_BASE="/tmp/${_USER}"

mkdir -p "${_BASE}/uv-cache" "${_BASE}/torchinductor" "${_BASE}/triton"

export UV_CACHE_DIR="${_BASE}/uv-cache"
export UV_PROJECT_ENVIRONMENT="${_BASE}/cs336-a2-venv"
export TORCHINDUCTOR_CACHE_DIR="${_BASE}/torchinductor"
export TRITON_CACHE_DIR="${_BASE}/triton"
# Prefer project-local tooling
export PATH="${HOME}/.local/bin:${PATH}"

cd "${_ROOT}"

echo "[cs336-a2] UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "[cs336-a2] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "[cs336-a2] cwd=${_ROOT}"

if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  echo "[cs336-a2] venv missing on this node — run: uv sync"
fi
