#!/usr/bin/env bash
# Run on the online preparation machine. Installs dependencies, NEVER builds
# DeepInfer/vendor operators and NEVER runs a model or requires a GPU.
set -euo pipefail
export VIRTUAL_ENV=/opt/deepinfer-venv
export PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
export DEEPINFER_ACCELERATOR=nvidia CC=gcc-13 CXX=g++-13 CUDA_HOME=/usr/local/cuda
cd /opt/DeepInfer
test "$(git rev-parse HEAD)" = a679bb86ce36e883a98a97733492b686e5b9a08e
git diff --exit-code
if git submodule status --recursive | grep -qE '^[-+U]'; then
  echo 'Incomplete or incorrect recursive source tree' >&2; exit 1
fi
python -m pip install -r requirements-nvidia.txt
python -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3
mkdir -p /opt/build-provenance
python -m pip freeze --all > /opt/build-provenance/pip-freeze.txt
if ! python -m pip check > /opt/build-provenance/pip-check.txt; then
  python - <<'PY'
from pathlib import Path
lines=Path('/opt/build-provenance/pip-check.txt').read_text().splitlines()
assert len(lines)==1 and lines[0].startswith('torch 2.13.0+cu132 has requirement nvidia-nccl-cu13==2.29.7'), lines
assert 'but you have nvidia-nccl-cu13 2.32.3' in lines[0], lines
Path('/opt/build-provenance/known-nccl-override.txt').write_text(
    'Official DeepInfer build.sh overrides NCCL 2.29.7 with 2.32.3; all other pip conflicts are rejected.\n')
PY
fi
git rev-parse HEAD > /opt/build-provenance/deepinfer.commit
git submodule status --recursive > /opt/build-provenance/submodules.txt
nvcc --version > /opt/build-provenance/nvcc.txt
gcc-13 --version > /opt/build-provenance/gcc.txt
dpkg-query -W -f='${binary:Package}\t${Version}\n' > /opt/build-provenance/os-packages.tsv
# A clean / empty HF cache is intentional; weights are supplied read-only later.
python - <<'PY'
import json,pathlib,torch
assert torch.__version__=='2.13.0+cu132' and torch.version.cuda=='13.2'
assert torch._C._GLIBCXX_USE_CXX11_ABI
pathlib.Path('/opt/build-provenance/buildkit.json').write_text(json.dumps({
    'stage':'offline_buildkit', 'platform':'linux/amd64',
    'deepinfer_compiled':False, 'vendor_operators_compiled':False,
    'gpu_tests_run':False, 'intended_build_host':'B300 SM103',
    'torch':torch.__version__,'cuda':torch.version.cuda,
    'source_commit':'a679bb86ce36e883a98a97733492b686e5b9a08e'
},indent=2)+'\n')
PY
# Keep compiler, source, JIT headers and installed build tools. These caches are
# not needed for the offline compile and would unnecessarily enlarge the image.
python -m pip cache purge
cat > /opt/deepinfer-entrypoint.sh <<'ENTRYPOINT'
#!/bin/bash
set -euo pipefail
export DEEPINFER_ACCELERATOR=nvidia VIRTUAL_ENV=/opt/deepinfer-venv
export PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="/opt/DeepInfer${PYTHONPATH:+:$PYTHONPATH}"
cd /opt/DeepInfer
exec "$@"
ENTRYPOINT
chmod 755 /opt/deepinfer-entrypoint.sh
echo BUILD_KIT_PREPARATION_COMPLETED_NO_ENGINE_COMPILE
