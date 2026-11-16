#!/usr/bin/env bash
# One entry point for the future B300 operator. No network / GPU actions occur
# unless the corresponding subcommand is explicitly invoked.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "${CONFIG:-}" ]]; then set -a; source "$CONFIG"; set +a; fi
RUNTIME="${RUNTIME:-nerdctl}"
WORK_DIR="${WORK_DIR:-/data/deepinfer-vllm-b300}"
DATA_ROOT="${DATA_ROOT:-/data/share/models/_datasets}"
case "$RUNTIME" in
  docker) RT=(docker) ;;
  nerdctl) RT=(nerdctl --namespace "${RUNTIME_NAMESPACE:-deepinfer-b300-bench}") ;;
  *) echo 'RUNTIME must be docker or nerdctl' >&2; exit 2 ;;
esac
run() { if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi; }
pinned() { [[ "$1" =~ @sha256:[a-f0-9]{64}$ ]] || { echo 'A verified manifest digest is required' >&2; exit 2; }; }
host_check() {
  [[ "${DRY_RUN:-0}" == 1 ]] && return
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'This kit requires Linux x86_64; do not emulate it on ARM' >&2; exit 2; }
  command -v "$RUNTIME" >/dev/null
  command -v python3 >/dev/null
}
gpu_check() {
  : "${GPU_IDS:?Set the explicitly allocated GPU indices or UUIDs}"
  [[ "$GPU_IDS" =~ ^[A-Za-z0-9,-]+$ ]] || { echo 'Invalid GPU list' >&2; exit 2; }
  [[ "${DRY_RUN:-0}" == 1 ]] && return
  python3 - "$GPU_IDS" <<'PY'
import csv,subprocess,sys
selected=sys.argv[1].split(',')
assert len(selected)==len(set(selected)), 'Duplicate GPU selection'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,compute_cap','--format=csv,noheader'],text=True)
rows=[[v.strip() for v in r] for r in csv.reader(raw.splitlines())]
uuids=set()
for value in selected:
 matches=[r for r in rows if value in (r[0],r[1])]
 assert len(matches)==1 and matches[0][3]=='10.3', 'GPU missing or not B300/SM103: '+value
 uuids.add(matches[0][1])
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert not any(r and r[0].strip() in uuids for r in csv.reader(raw.splitlines())), 'Selected GPU has an existing compute process'
print('Selected GPUs are SM103 and have no compute processes; verify scheduler allocation separately.')
PY
}
new_dir() { if [[ "${DRY_RUN:-0}" != 1 ]]; then test ! -e "$1"; mkdir -p "$1"; fi; }
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
case "${1:-plan}" in
  plan)
    printf '%s\n' '1 pull/load  2 data/check  3 build  4 freeze-model/smoke  5 publish/export  6 gates/probe  7 g3  8 bench'
    printf '%s\n' 'DeepInfer build runs on B300 with --network none. GPU_IDS is never defaulted to all.' ;;
  prepare)
    host_check; : "${BUILD_CONTEXT:?Choose a new directory on the ONLINE preparation machine}" "${BUILD_KIT_TAG:?Choose a new build-kit image tag}"
    run bash "$ROOT/scripts/prepare_build_context.sh" "$BUILD_CONTEXT"
    run "${RT[@]}" build --platform linux/amd64 -t "$BUILD_KIT_TAG" "$BUILD_CONTEXT"
    echo 'Publish the kit, verify its manifest digest, and set BUILD_KIT_IMAGE before the B300 build stage.' ;;
  pull)
    host_check
    for image in "${BUILD_KIT_IMAGE:?}" "${VLLM_IMAGE:?}" "${DATA_IMAGE:?}"; do pinned "$image"; run "${RT[@]}" pull "$image"; done ;;
  load)
    host_check
    : "${ARCHIVE_DIR:?Directory containing image archives and SHA256SUMS}"
    if [[ "${DRY_RUN:-0}" != 1 ]]; then (cd "$ARCHIVE_DIR"; sha256sum -c SHA256SUMS); fi
    shopt -s nullglob
    archives=("$ARCHIVE_DIR"/*.tar)
    [[ ${#archives[@]} -gt 0 ]] || { echo 'No image archives' >&2; exit 2; }
    for f in "${archives[@]}"; do run "${RT[@]}" load -i "$f"; done ;;
  check) host_check; gpu_check ;;
  data)
    host_check; pinned "${DATA_IMAGE:?}"
    run "${RT[@]}" run --rm --network none -v "$DATA_ROOT:/target" "$DATA_IMAGE" sh -ec \
      'test ! -e /target/b300-source-subset-v1; test ! -e /target/b300-paired-text-v1; test ! -e /target/b300-http-v1; test ! -e /target/model-metadata; cp -a /datasets/. /target/; cd /target/b300-source-subset-v1; sha256sum -c SHA256SUMS' ;;
  build)
    host_check; gpu_check; pinned "${BUILD_KIT_IMAGE:?}"
    : "${DI_LOCAL_IMAGE:?Set a new local output image tag}"
    out="$WORK_DIR/build-$stamp"; new_dir "$out"
    container="deepinfer-b300-build-$stamp"
    cmd=("${RT[@]}" run --name "$container" --network none --gpus "\"device=$GPU_IDS\"" -e "NVIDIA_VISIBLE_DEVICES=$GPU_IDS"
      --cpus "${BUILD_CPUS:-16}" --memory "${BUILD_MEMORY:-96g}" --shm-size 8g
      -e "MAX_JOBS=${MAX_JOBS:-8}" -v "$ROOT:/benchkit:ro" --entrypoint /bin/bash
      "$BUILD_KIT_IMAGE" /benchkit/scripts/build_engine_offline.sh)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then run "${cmd[@]}"; else
      if "${RT[@]}" image inspect "$DI_LOCAL_IMAGE" >/dev/null 2>&1; then echo 'Output image already exists; choose a new tag' >&2; exit 2; fi
      "${cmd[@]}" 2>&1 | tee "$out/build.log"
    fi
    run "${RT[@]}" commit --change 'ENTRYPOINT ["/opt/deepinfer-entrypoint.sh"]' --change 'CMD ["bash"]' "$container" "$DI_LOCAL_IMAGE"
    run "${RT[@]}" cp "$container:/opt/build-provenance" "$out/provenance"
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      "${RT[@]}" image inspect "$DI_LOCAL_IMAGE" > "$out/image-inspect.json"
      echo 'Compiled image created. This does not pass model or performance gates.'
    fi ;;
  freeze-model)
    host_check; : "${MODEL_DIR:?}"
    run python3 "$ROOT/scripts/freeze_model.py" --model "$MODEL_DIR" \
      --revision 08c8657f02291e11b6f6b63e0e16014f8e748518 --output "${MODEL_MANIFEST:?Choose a new manifest path}" ;;
  smoke)
    host_check; gpu_check
    [[ "$(awk -F, '{print NF}' <<< "$GPU_IDS")" -eq 8 ]] || { echo 'This official EP8/CP8 smoke candidate requires 8 allocated GPUs' >&2; exit 2; }
    : "${MODEL_DIR:?}" "${DI_LOCAL_IMAGE:?}" "${MODEL_MANIFEST:?Complete and review model hashes first}"
    [[ "${DRY_RUN:-0}" == 1 || -s "$MODEL_MANIFEST" ]] || { echo 'Missing model manifest' >&2; exit 2; }
    out="$WORK_DIR/smoke-$stamp"; new_dir "$out"
    cmd=("${RT[@]}" run --name "deepinfer-b300-smoke-$stamp" --network none --gpus "\"device=$GPU_IDS\"" -e "NVIDIA_VISIBLE_DEVICES=$GPU_IDS"
      --ipc host --ulimit memlock=-1:-1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash-1017:ro" "$DI_LOCAL_IMAGE"
      python examples/offline_inference.py --ep 8 --cp 8 --dp 1
      --model_name /models/DeepSeek-V4.1-Flash-1017 --check_structured_output)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then run "${cmd[@]}"; else "${cmd[@]}" 2>&1 | tee "$out/server.log"; fi ;;
  publish)
    host_check; : "${DI_LOCAL_IMAGE:?}" "${DI_SWR_TAG:?Use a new SWR tag}"
    run "${RT[@]}" tag "$DI_LOCAL_IMAGE" "$DI_SWR_TAG"
    run "${RT[@]}" push "$DI_SWR_TAG"
    echo 'Record the verified remote manifest digest as DI_IMAGE before formal benchmarking.' ;;
  export)
    host_check; : "${DI_LOCAL_IMAGE:?}" "${EXPORT_ARCHIVE:?Choose a new .tar path}"
    [[ "${DRY_RUN:-0}" == 1 || ! -e "$EXPORT_ARCHIVE" ]] || { echo 'Archive exists' >&2; exit 2; }
    run "${RT[@]}" save -o "$EXPORT_ARCHIVE" "$DI_LOCAL_IMAGE"
    if [[ "${DRY_RUN:-0}" != 1 ]]; then sha256sum "$EXPORT_ARCHIVE" > "$EXPORT_ARCHIVE.sha256"; fi ;;
  gates)
    host_check; : "${ENGINE:?}" "${RECIPE_FILE:?}" "${GATES_FILE:?Choose a new gate record path}"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then echo 'Would create a gate template with all approval/pass flags false'; else
      python3 - "$ROOT" "$ENGINE" "$RECIPE_FILE" "$DATA_ROOT" "${SCENARIO:-s8k}" "$GATES_FILE" <<'PY'
import json,sys,hashlib,pathlib
root,engine,recipe,data,scene,out=sys.argv[1:]
assert engine in ('deepinfer','vllm')
recipe=pathlib.Path(recipe);data=pathlib.Path(data)/'b300-paired-text-v1'
g={'resource_authorized':False,'recipe_approved':False,'G0':False,'G1':False,'G2':False,'G3':False,'evidence_directory':'FILL_WITH_REVIEWED_EVIDENCE'}
g.update(engine=engine,recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),G3=False,
  allowed_request_sha256=[hashlib.sha256((data/n).read_bytes()).hexdigest() for n in [scene+'.jsonl','warm-sharegpt.jsonl']],
  warmup_path='/datasets/b300-paired-text-v1/warm-'+scene+'.jsonl',
  warmup_sha256=hashlib.sha256((data/('warm-'+scene+'.jsonl')).read_bytes()).hexdigest())
p=pathlib.Path(out);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as f:json.dump(g,f,indent=2);f.write('\n')
print('Created template. Review actual G0/G1/G2 evidence and authorization; no gate was marked passed.')
PY
    fi ;;
  bench|g3|probe)
    mode="$1"
    host_check; gpu_check
    : "${ENGINE:?deepinfer or vllm}" "${GATES_FILE:?}" "${MODEL_DIR:?}" "${POINT_ID:?}" "${RECIPE_FILE:?}"
    case "$ENGINE" in deepinfer) image="${DI_IMAGE:?}"; entry=(--entrypoint /opt/deepinfer-entrypoint.sh) ;; vllm) image="${VLLM_IMAGE:?}"; entry=(--entrypoint /bin/bash) ;; *) echo 'Unknown engine' >&2; exit 2 ;; esac
    pinned "$image"
    [[ "$POINT_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid point ID' >&2; exit 2; }
    if [[ "$mode" == bench && "${DRY_RUN:-0}" != 1 ]]; then python3 - "$GATES_FILE" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]));assert g.get('G3') is True, 'G3 evidence required before performance stage'
PY
    fi
    if [[ "$mode" == g3 ]]; then BATCH_SIZE=1; REQUEST_COUNT=24; SCENARIO=s8k; fi
    if [[ "$mode" == probe ]]; then BATCH_SIZE=1; REQUEST_COUNT=1; SCENARIO=warm-sharegpt; fi
    out="$WORK_DIR/results/$POINT_ID"; new_dir "$out"
    args=(python3 /benchkit/scripts/native_batch.py --engine "$ENGINE" --recipe /recipe.json
      --requests "/datasets/b300-paired-text-v1/${SCENARIO:-s8k}.jsonl" --batch-size "${BATCH_SIZE:-8}"
      --limit "${REQUEST_COUNT:-128}" --gates-record /gates.json --output /results/point)
    if [[ "$mode" == g3 ]]; then args+=(--g3); fi
    if [[ "$mode" == probe ]]; then args+=(--probe); fi
    if [[ "$ENGINE" == vllm ]]; then args=(-c 'exec "$@"' bash "${args[@]}"); fi
    cmd=("${RT[@]}" run --name "bench-$POINT_ID" --network none --gpus "\"device=$GPU_IDS\"" --ipc host
      -e "NVIDIA_VISIBLE_DEVICES=$GPU_IDS" -e "BENCH_IMAGE_MANIFEST_REF=$image"
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -v "$ROOT:/benchkit:ro" -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash-1017:ro"
      -v "$DATA_ROOT:/datasets:ro" -v "$GATES_FILE:/gates.json:ro" -v "$RECIPE_FILE:/recipe.json:ro"
      -v "$out:/results" "${entry[@]}" "$image" "${args[@]}")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then run "${cmd[@]}"; else "${cmd[@]}" 2>&1 | tee "$out/client.log"; fi ;;
  *) echo 'Usage: b300.sh plan|prepare|pull|load|check|data|build|freeze-model|smoke|publish|export|gates|probe|g3|bench' >&2; exit 2 ;;
esac
