#!/usr/bin/env bash
# FUTURE operator command. Online source preparation only; no GPU/model actions.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
context="${1:?Usage: prepare_build_context.sh <new-context-directory>}"
test ! -e "$context"
mkdir -p "$context/source/DeepInfer" "$context/scripts"
repo="$context/source/DeepInfer"
git -C "$repo" init
git -C "$repo" remote add origin https://github.com/deepseek-ai/DeepInfer.git
git -C "$repo" fetch --depth 1 origin a679bb86ce36e883a98a97733492b686e5b9a08e
git -C "$repo" checkout --detach FETCH_HEAD
git -C "$repo" submodule update --init --recursive --depth 1
git -C "$repo" diff --exit-code
git -C "$repo" submodule status --recursive > "$context/source-submodules.txt"
if grep -qE '^[-+U]' "$context/source-submodules.txt"; then exit 1; fi
cat > "$context/Dockerfile" <<'DOCKERFILE'
ARG CUDA_BASE=swr.cn-north-4.myhuaweicloud.com/runner/cuda@sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d
FROM ${CUDA_BASE}
ENV DEBIAN_FRONTEND=noninteractive VIRTUAL_ENV=/opt/deepinfer-venv DEEPINFER_ACCELERATOR=nvidia
ENV PATH=/opt/deepinfer-venv/bin:/usr/local/cuda/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES=void
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev gcc-13 g++-13 git ca-certificates \
    ninja-build cmake make pkg-config libnuma-dev libibverbs-dev librdmacm-dev libdw-dev libelf-dev binutils \
    && python3.12 -m venv /opt/deepinfer-venv \
    && python -m pip install --upgrade pip setuptools wheel packaging
COPY source/DeepInfer /opt/DeepInfer
COPY scripts/prepare_buildkit_inside.sh /prepare-buildkit.sh
RUN bash /prepare-buildkit.sh
WORKDIR /opt/DeepInfer
ENTRYPOINT ["/opt/deepinfer-entrypoint.sh"]
CMD ["bash"]
DOCKERFILE
cp "$root/scripts/prepare_buildkit_inside.sh" "$context/scripts/"
echo 'Context prepared. Building/publishing the kit is a separate explicit command.'
