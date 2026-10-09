接口适配：当前容器的 vLLM API server 位于 entrypoints/openai/api_server.py。仅替换导入路径及CLI启动入口，保留上游的prompt、LoRA、decision head偏置、温度与readout逻辑。初测候选不超过16项，使用上游allowed_token_ids readout路径，不要求缺失的logprob_token_ids字段。初测enforce-eager，不声称图模式开启。


### Candidate probability readout (attempt 05 invalidated)

The installed Ascend model runner creates `AscendSampler()` with its default
`raw_logprobs`; it does not pass the engine's configured `processed_logprobs`.
Consequently, top-K includes unrelated vocabulary tokens before the candidate
mask, even though generation itself obeys the candidate allowlist. The upstream
server silently substitutes -1e9 for absent candidates. This can yield artificial
one-hot distributions and invalidates attempt-05 capability outputs.

A raw completion requesting IDs 32/33/34 returned IDs 32/357/849. Adding a
common +100 logit bias did not fix the returned logprobs. Attempt 06 therefore
patches the sampler constructor's default in this isolated server process to
`processed_logprobs`, preserving the upstream candidate mask, decision bias and
calibration formula. No installed engine files or model weights are changed.
The server now rejects missing or nonfinite candidate logprobs. Before replay,
the raw API must return every requested candidate, no unrelated token, and
exponentiated logprobs summing to one within 0.001, for 3/20 Choice labels,
Noul, and the six Score verbalizers. Correctness and concurrent probability
parity checks then follow. See `readout-amendment.json` for the frozen gate.

The model publisher identifies Qwen3.8-27B as the base. All 18 base shard hashes
match that repository at the revisions in `base-weight-provenance.json`.
Internal `qwen3_5` architecture class names describe the compatible implementation;
they do not replace the publisher's base model provenance.


### Candidate mask row swap (attempt 06 invalidated)

The installed upstream `InputBatch.swap_states` uses tuple assignment on tensor
row views for `allowed_token_ids_mask_cpu_tensor`. The first assignment mutates
the view used by the second assignment, so both rows become the original second
row. `NPUInputBatch` inherits this implementation. Request reordering can therefore
apply a different request's candidate mask. The full replay exposed 20 text and
6 visual HTTP failures, which the strict readout checks caught. All attempt-06
results are excluded; the performance controller stopped before any performance
run.

`mask-swap-reproduction.json` records a reproduction extracted from the installed
source and its SHA. Attempt 07 wraps `InputBatch.swap_states` in the isolated
server: save independent copies of the two mask rows, call the original method,
then restore the correctly exchanged mask rows. Other batch state handling,
async scheduling, NPU kernels, weights and calibration parameters are unchanged.
The first actual invocation is logged in each worker. The correctness gate now
adds 132 mixed image/text requests, disjoint candidate sets and varied prompt
lengths, checked against sequential reference distributions with maximum absolute
probability difference <=0.05. The complete frozen ability replay is repeated.
