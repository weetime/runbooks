#!/usr/bin/env bash
# Runs INSIDE the prepared image, on B300, with networking disabled by b300.sh.
set -euo pipefail
export VIRTUAL_ENV=/opt/deepinfer-venv DEEPINFER_ACCELERATOR=nvidia
export PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH=/opt/DeepInfer CC=gcc-13 CXX=g++-13 CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST='10.3a' MAX_JOBS="${MAX_JOBS:-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
cd /opt/DeepInfer
test "$(uname -m)" = x86_64
test "$(git rev-parse HEAD)" = a679bb86ce36e883a98a97733492b686e5b9a08e
git diff --exit-code
if git submodule status --recursive | grep -qE '^[-+U]'; then
  echo 'Source tree incomplete: rebuild / restore the preparation kit' >&2; exit 1
fi
python - <<'PY'
import importlib.metadata as m,torch
assert torch.__version__=='2.13.0+cu132' and torch.version.cuda=='13.2'
assert m.version('nvidia-nccl-cu13')=='2.32.3'
assert torch.cuda.is_available(), 'No GPU exposed to build container'
assert all(torch.cuda.get_device_capability(i)==(10,3) for i in range(torch.cuda.device_count())), 'Requires B300/SM103'
PY
# These are the two compile steps of the pinned official build.sh. Its pip
# steps have already been performed in the buildkit, so no online pip step runs.
bash scripts/build_vendor.sh
bash scripts/build.sh build
git diff --exit-code
git submodule foreach --recursive 'git diff --exit-code'
python -m pip freeze --all > /opt/build-provenance/engine-pip-freeze.txt
find /opt/DeepInfer/build -name '*.whl' -exec sha256sum {} \; > /opt/build-provenance/wheels.sha256
python - <<'PY'
import json,pathlib,torch,importlib.util
assert importlib.util.find_spec('deepinfer._C') is not None
pathlib.Path('/opt/build-provenance/engine-build.json').write_text(json.dumps({
    'stage':'compiled_on_b300', 'source_commit':'a679bb86ce36e883a98a97733492b686e5b9a08e',
    'devices':[{'name':torch.cuda.get_device_name(i),'cc':list(torch.cuda.get_device_capability(i))}
               for i in range(torch.cuda.device_count())],
    'model_tests_run':False,'performance_tests_run':False,
    'note':'Compilation is not model correctness or performance acceptance'
},indent=2)+'\n')
PY
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
echo DEEPINFER_B300_OFFLINE_COMPILE_COMPLETED
