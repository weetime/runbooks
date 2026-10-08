---
license: apache-2.0
language:
- en
pretty_name: Typed Decisions
size_categories:
- n<1K
task_categories:
- text-classification
tags:
- structured-decisions
- calibration
- probabilistic-classification
- system-one
- workflow-evaluation
- synthetic
configs:
- config_name: agent_trace_observability
  data_files:
  - split: train
    path: agent_trace_observability/train-*.parquet
  - split: test
    path: agent_trace_observability/test-*.parquet
- config_name: customer_service
  data_files:
  - split: train
    path: customer_service/train-*.parquet
  - split: test
    path: customer_service/test-*.parquet
- config_name: invoice_processing
  data_files:
  - split: train
    path: invoice_processing/train-*.parquet
  - split: test
    path: invoice_processing/test-*.parquet
- config_name: security_incidents
  data_files:
  - split: train
    path: security_incidents/train-*.parquet
  - split: test
    path: security_incidents/test-*.parquet
- config_name: all
  data_files:
  - split: train
    path: all/train-*.parquet
  - split: test
    path: all/test-*.parquet
---

# Typed Decisions

A benchmark for typed probabilistic decisions. A model gets one piece of
unstructured state and answers five typed questions about it at once, and every
answer is a probability distribution, not a single label.

The schema follows the System One primitives (`noul`, `choice`, `score`) used by
[TypeSafe AI](https://typesafe.ai), so a row replays against any API with that
shape. The benchmark is independent: it is not affiliated with TypeSafe and does
not reproduce their Jev model.

![Accuracy against KL divergence and against latency for every scored model](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/assets/leaderboard.png)

## Leaderboard

`test` split: 400 cases, 2,000 decisions. This table lists general models scored
zero-shot: they have never seen these workflows or their question schemas.
Models that were fitted or fine-tuned on `train` are in the second table below;
the two tables are not comparable.

| # | Model | Kind | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ | p50 latency | Price / 1M input |
|---|---|---|---|---|---|---|---|---|
| 1 | **[meraGPT Decider 1](https://meragpt.com/models/state-decider-1?utm_source=huggingface&utm_medium=dataset_card&utm_campaign=typed-decisions)** (`sd-1`) | general, zero-shot | **0.768** | **0.096** | **0.052** | 0.180 | **526 ms** | **$0.03** |
| 2 | [Liquid AI d1](https://docs.liquid.ai/lfm/models/decision-models) (`d1:free`) | general, zero-shot | 0.742 | 0.475 | 0.155 | 0.124 | 525 ms | free tier |
| 3 | TypeSafe Jev 1.13.0 | general, zero-shot | 0.727 | 1.442 | 0.148 | 0.144 | 710 ms | $0.042 |
| 4 | [Featherless Simple Jev](https://simple-jev.featherless.ai) (`Qwen3.6-35B-A3B-classifier`) | general, zero-shot | 0.716 | 0.488 | 0.176 | – | – | free demo |
| 5 | [prima-ratio + 12B](https://github.com/andrea-tomassi/prima-ratio) § | general, zero-shot | 0.702 | 0.564 | 0.234 | 0.146 | 700 ms‡ | local GGUF |
| 6 | [OpenDecider-small](https://huggingface.co/manjunathshiva/opendecider-small) § | general, zero-shot, open weights | 0.671 | 0.211 | 0.117 | – | 40 ms‡ | open weights |
| 7 | [Bongard-mini](https://huggingface.co/AgentBull/bongard-mini) § | general, zero-shot, open weights | 0.594 | 0.256 | 0.132 | 0.067 | 225 ms‡ | open weights |
| 8 | [Jeff-Gemma4-E2B](https://huggingface.co/mstrasser/Jeff-Gemma4-E2B) | general, zero-shot, open weights | 0.561 | 0.403 | 0.219 | 0.188 | 2,272 ms† | open weights |
| 9 | [Jeff-Qwen3.5-2B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-2B) | general, zero-shot, open weights | 0.511 | 0.460 | 0.237 | 0.203 | 1,346 ms† | open weights |
| 10 | [Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B) | general, zero-shot, open weights | 0.483 | 0.679 | 0.313 | 0.251 | 662 ms† | open weights |
| – | Prior (ignores the input) | reference | 0.470 | 0.347 | 0.189 | 0.088 | – | – |
| – | Uniform (same probability on every option) | reference | 0.308 | 0.444 | 0.238 | 0.169 | – | – |

Every row was scored by sending the whole case (state plus all five questions) in
one request. Request shape matters: in a third-party run Jev's yes/no accuracy was
0.843 with one question per request and 0.788 alongside the others.

Hosted rows are p50 end to end from a client, one request at a time.
† Measured on the same machine as the model (M3 Max), not comparable with hosted latency.
‡ Reported by the submitter on their own hardware (a different GPU and a different unit for each), not comparable with the other rows.
§ Self-reported by the submitter in the linked discussion and not re-run by us. Metric definitions follow the submitter's report; ECE in particular is computed differently by different submitters.

### Fitted or fine-tuned on `train`

These models were trained on this benchmark's `train` split, so they are not
zero-shot and their scores are not comparable with the table above. All were
self-reported in the linked discussions and were not re-run by us, except the two
specialists marked ours and the three Bekko rows, which we scored ourselves. Scores well above the 0.735 teacher self-agreement mean a
model is learning the teacher's quirks.

| Model | Kind | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ | Reported latency |
|---|---|---|---|---|---|---|
| [OpenDecider-large-td](https://huggingface.co/manjunathshiva/opendecider-large-td) (Qwen3-Next-80B-A3B + LoRA) | general, then fine-tuned on `train` | 0.801 | 0.081 | 0.044 | – | 440 ms‡ |
| [od1-typed-decisions](https://huggingface.co/mvbalaji/od1-typed-decisions) (Qwen3.5-4B) | specialist | 0.797 | 0.082 | 0.045 | 0.156¶ | – |
| [OpenDecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano) (400M encoder) | general, then fine-tuned on `train` | 0.796 | 0.079 | 0.043 | – | 17 ms‡ |
| [OpenDecider-small-td](https://huggingface.co/manjunathshiva/opendecider-small-td) (Qwen3-4B + LoRA) | general, then fine-tuned on `train` | 0.792 | 0.080 | 0.043 | – | 40 ms‡ |
| [OpenDecider-medium-td](https://huggingface.co/manjunathshiva/opendecider-medium-td) (Qwen3-30B-A3B + LoRA) | general, then fine-tuned on `train` | 0.788 | 0.081 | 0.044 | – | 214 ms‡ |
| [soft-decider-421m](https://huggingface.co/winwinwinbb/soft-decider-421m) | specialist | 0.774 | – | – | 0.141¶ | 50 ms‡ |
| [Laya typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | fitted on `train` | 0.766 | – | – | – | – |
| [Bekko System One v0 400M](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m), scored by us | trained on `train` and other data | 0.668 | 0.214 | 0.113 | 0.143 | 808 ms† |
| ModernBERT-base (149M), ours | specialist, fitted per workflow | 0.646 | 0.223 | 0.119 | 0.179 | 349 ms† |
| MiniLM-L6 (22M), ours | specialist, fitted per workflow | 0.587 | 0.262 | 0.143 | 0.108 | 22 ms† |
| [Bekko System One v0 68M](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m), scored by us | trained on `train` and other data | 0.537 | 0.293 | 0.160 | 0.116 | 176 ms† |
| [Bekko System One v0 17M](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m), scored by us | trained on `train` and other data | 0.483 | 0.344 | 0.203 | 0.136 | 33 ms† |

¶ The submitter's own ECE definition, which does not match the one used above.

**[meraGPT Decider 1](https://meragpt.com/models/state-decider-1?utm_source=huggingface&utm_medium=dataset_card&utm_campaign=typed-decisions) is state of
the art among zero-shot models on this benchmark.** It leads Jev on every question type (`noul` 0.840 vs
0.775, `choice` 0.733 vs 0.720, `score` 0.739 vs 0.696), its distributions sit far
closer to the gold (KL 0.096 vs 1.442), it is faster end to end, and it costs less
per token. It answers `POST /v1/systemone` at
[meragpt.com](https://meragpt.com/docs/systemone?utm_source=huggingface&utm_medium=dataset_card&utm_campaign=typed-decisions), so the typesafe-sdk works
against it by setting `TYPESAFE_BASE_URL`.

To add a model, score it on `test` with the full distributions and open a
discussion with the numbers and the mode (specialist or general) it used.

### Notes on the rows

- **Liquid AI d1** was measured on 2026-09-30 through Liquid's API
  (`https://api.liquid.ai/v1/systemone`, `model: d1:free`), with the same client
  code as the Jev row: all 2,000 decisions, zero errors. It ties Decider 1 on
  `noul` (0.840) and `choice` (0.732 vs 0.733) and trails on `score` (0.677 vs
  0.739), and it beats Jev on accuracy and KL. Liquid lists no per-token price
  yet, so the row shows the free tier.
- **Jev 1.13.0** was measured on 2026-09-18 through TypeSafe's API (`jev-latest`,
  which reported `jev-1.13.0`): all 2,000 decisions, zero errors, $0.016 in total.
  Its accuracy is near the 0.735 ceiling, but it puts nearly all its probability
  on one answer, which is where the KL gap comes from. Its confidence is not badly
  calibrated (overconfidence +0.023); the gold is a three-sample spread it does not
  reproduce.
- **Jeff** ([firelex/jeff](https://github.com/firelex/jeff), Apache-2.0) was scored
  on 2026-09-29 through the same client code as Jev, against Jeff's own
  `jeff-serve` (commit `2c1bfce`) with the calibration each checkpoint ships. All
  three clear the Prior on accuracy but not on KL. Jeff's own 83.1% comes from a
  different five-benchmark panel. If there is a better way to serve them, open a
  discussion and we will rescore.
- **Self-reported rows** (§ above, and the second table) come from discussions
  [#5](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/5)
  (prima-ratio), [#7](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/7)
  (Bongard-mini), [#8](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/8)
  (OpenDecider), [#4](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/4)
  (od1), [#3](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/3)
  (soft-decider) and [#2](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/2)
  (Laya). Each submitter states the mode; we have not re-run them. If a number looks
  wrong, say so in the discussion and we will correct it.
- **Bekko System One v0** ([hotchpotch](https://huggingface.co/collections/hotchpotch/bekko-system-one-v0-6abc47eab9c4fe4ec83d9e30)) was scored by us on 2026-10-01 on CPU (M3 Max, one call per case with all five questions, pinned revisions `b886a1f9` 17M, `ab7685f2` 68M, `1960df56` 400M), with the same scorer as the other rows. Its training data includes this benchmark's `train` split, with our teacher labels, so it sits in the second table; none of our 400 test cases are in its training data. We mapped our option and rubric text onto its candidate format in our option order. Its model cards say the license is not yet finalized. The author invited a score on X; we are glad to re-score with a different input mapping if he prefers.
- **Specialists** use [Adaptive Classifier](https://github.com/codelion/adaptive-classifier)
  0.2.0, one classifier per question on a frozen encoder, tuned on a held-out
  quarter of `train` (mean pooling, `max_length` 512, 30 epochs,
  `prototype_weight` 0.3). Each case is entered four times, split across labels in
  proportion to its gold, so the soft target survives hard-label training; that
  cut KL by a third.
- **Prior** answers each question's `train` label frequencies for every case. It
  has the best ECE while knowing nothing, which is why KL and Brier are the
  columns to read, not ECE.

## Reading a score

Gold is the mean of three samples from a teacher of roughly 4B-class capability,
so a score measures agreement with that teacher, not correctness. A better model
can score lower wherever the teacher is wrong (it missed a duplicate invoice whose
ID matched an earlier one).

| Reference | Accuracy | What it is |
|---|---|---|
| Prior | 0.470 | the floor: label frequencies, ignoring the input |
| Perfect factor recovery | 0.704 | a model fitted to the latent factors that generated each case |
| Teacher self-agreement | 0.735 | a fresh teacher sample against gold built from the others |

Scores well above 0.735 mean a model is learning the teacher's quirks. Per-question
ceilings vary from 0.560 (`agent_trace/urgency`) to 0.937
(`customer_service/category`), so read scores per question as well as on average.

**Specialist and general numbers are not comparable.** A specialist is fitted on
`train` for these four workflows and cannot answer anything else. A general model
takes any question schema at request time and has never seen these. Say which
mode you used; the gap between them is the price of generality, not a quality
ranking.

## The data

| Type | Answer | Shape |
|---|---|---|
| `noul` | yes/no | probability that the statement is true |
| `choice` | one of N labels | distribution over labels, plus confidence |
| `score` | an ordered rubric | distribution over levels, plus an expected score |

Every option carries a written description in `criteria`, and the descriptions
are part of the input.

| Workflow | Decision | Train | Test |
|---|---|---|---|
| `agent_trace_observability` | Does an agent run need human review, and how urgently? | 300 | 100 |
| `customer_service` | The right response and action for a customer thread and account. | 300 | 100 |
| `invoice_processing` | Pay, hold or reject a vendor bill against its order and delivery. | 300 | 100 |
| `security_incidents` | Close, investigate or contain a security alert, given machine history. | 300 | 100 |

`state` and `questions` together are exactly the body of a `POST /v1/systemone`
request. `gold` holds the full gold distributions, and flat
`<question>__label` / `__probabilities` / `__score` / `__probability_true` columns
hold the same answers for convenience. `factors` and `label_agreement` describe how
the case was built and are not model input.

```python
from datasets import load_dataset
import json

ds = load_dataset("LocalLLaMA/typed-decisions", "customer_service", split="test")
row = ds[0]
state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
print(gold["urgency"]["probabilities"])   # score against the full distribution
```

Report KL or log loss and Brier next to accuracy; calibration is the point.

## How it was built

Each case starts from independently sampled latent factors (topic, tone,
severity, discrepancy type and so on), which a model renders into free text where
the artefact is textual and keeps structured where it is structured. A teacher
labels each case three times at temperature 0.7, and the gold is the mean of those
distributions, so it stays soft where a decision is genuinely ambiguous. Before
release, an audit checks state diversity, label balance, and that the gold
actually tracks the input. `train` comes from a separate run at a different seed;
packaging refuses to build if any case id or state appears in both splits.

The write-up behind the benchmark, including the two bugs it exposed in the
classifier library: [Typed Decisions on Latent Node](https://latentnode.pages.dev/articles/typed-decisions.html).
