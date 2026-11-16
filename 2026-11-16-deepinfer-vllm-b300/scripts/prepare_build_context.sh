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
cp "$root/Dockerfile.buildkit" "$context/Dockerfile"
cp "$root/scripts/prepare_buildkit_inside.sh" "$context/scripts/"
echo 'Context prepared. Building/publishing the kit is a separate explicit command.'
