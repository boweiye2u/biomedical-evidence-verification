# Biomedical Evidence Verification v2
## Final Implementation Check, Paths, Configurations, and Environments

**Date:** 2026-10-08  
**Status:** Implementation checklist (revision 4, final)  
**Protocol of record:** `2026-10-08-biomedical-llm-finetuning-proposal-final-v4.md` + the three reviewer tweaks (Section 23). Commit it to `docs/posttraining/`. If this checklist and final-v4 disagree on design, final-v4 wins; on paths and commands, this checklist wins.  
**Repository:** `/home/boweiye2/Dropbox/Non-coding-variant/scientific-rag`

---

## 0. Change Log (revision 4, final)

| # | Fix | Sections |
|---|---|---|
| 1 | **M3 gets one separate, pre-declared held-out run** after its own freeze; the main held-out run (B0-B3, M1, M2) never reruns | 1, 15, 16, 20 |
| 2 | **Finalist selection rule:** pick the finalist by its 3-seed DEV mean; then test that finalist's seed 0 | 14 |
| 3 | **Stop condition aligned with final-v4 Section 3.2:** an unclear v1 scoring rule is resolved by the fallback rule plus re-scoring B0, not by stalling | 21 |
| 4 | **Run folders named by content,** so "m1" never means both a milestone and the gold-only LoRA model | 9, 11, 16, 17 |
| 5 | **Missing protocol items restored** (primary vs secondary metrics, success criteria, over-abstention check, calibration, rationale scope, stop threshold, three tweaks) | 14, 23 |
| 6 | **Tuned models are evaluated through the same vLLM serving stack as the baselines** | 11, 14 |
| 7 | **Few-shot context-length check** | 10, 11 |
| 8 | **Resume PDFs in repo** and **git-in-Dropbox** checks | 2, 10, 19 |
| 9 | **300-example split terminology corrected:** it is a fixed SciFact benchmark split previously evaluated in v1, not a pristine unseen test set for v2 | 1, 10, 14, 16, 20, 21, 23 |
| 10 | **Training schema separates original SciFact label from evidence-conditioned context label** | 12, 23 |
| 11 | **Benchmark-reuse disclosure:** v2's design was partly motivated by v1 error analysis on the 300-example split, so DEV results for the seed-0 checkpoint are reported alongside as corroboration; B0 reproduction/re-scoring is the only pre-freeze use of the benchmark | 1, 10, 23 |
| 12 | **Label fields defined per claim-abstract pair:** `original_scifact_label` → `claim_gold_label` + `pair_annotation`; Week 0 check for claims whose evidence abstracts carry different labels | 10, 12, 23 |
| 13 | **Evaluation labels renamed E1-E4** to eliminate the A-D collision with training conditions; all evaluation references updated, including over-abstention | 14, 23 |

---

## 1. Purpose

This document is the final implementation checklist for the LLM post-training extension of the existing biomedical retrieval and verification project.

The project does **not** rebuild the retrieval system. The intended progression is:

```text
Existing v1 system
    -> reproduce/freeze configuration
    -> establish B0-B3 baselines
    -> LoRA SFT: M1 gold-only
    -> LoRA SFT: M2 mixed evidence
    -> frozen benchmark evaluation (B0-B3, M1, M2)
    -> full fine-tuning: M3 + FSDP scaling
    -> one pre-declared benchmark run for M3 only
    -> tuned-model serving benchmark
```

Primary scientific question:

> Does task-specific post-training improve evidence-grounded verification, particularly when retrieval supplies imperfect evidence?

Secondary systems question:

> Does full fine-tuning provide enough benefit over LoRA to justify its cost, and how efficiently does that workload scale across GPUs?

### Benchmark status

The 300-example SciFact split is **not a pristine unseen test set for v2**, because it was already evaluated and analyzed in v1. In v2 it is treated as a fixed benchmark:

```text
fixed before v2 training
not used for v2 parameter fitting
not used for v2 checkpoint/model selection
previously evaluated in v1
```

All v2 model selection remains confined to TRAIN/CV and DEV. Except for Week 0 B0 reproduction/re-scoring, all new v2 model comparisons on the 300-example benchmark occur only after the v2 configuration is frozen.

**Benchmark-reuse disclosure (required in the README and report):**

- v2's design was partly motivated by v1 error analysis on this same split (the gold-document diagnostic and the observation that Qwen still erred with correct evidence). Mixed-evidence training targets failures observed on this benchmark, so v2 benchmark gains may be mildly optimistic. [INFERRED, MED-HIGH]
- Therefore report **DEV results for the seed-0 inferential checkpoint alongside** the benchmark results, as corroboration. DEV is also not pristine (it was used for finalist selection), so neither number is described as an unbiased generalization estimate.
- A genuinely unseen evaluation requires a new compatible external dataset (optional extension, Section 22).

**The only pre-freeze use of the benchmark** is Week 0 reproduction of B0 and, if needed, re-scoring B0 under the final-v4 3.2 fallback rule. Choose the scoring rule from its definition only, without comparing how alternative rules change any result.

---

## 2. Existing Repository Layout

Current repository:

```text
/home/boweiye2/Dropbox/Non-coding-variant/scientific-rag
```

Current top-level layout:

```text
scientific-rag/
├── 2026-09-30-biomedical-retrieval-project-proposal-v4.md
├── 2026-09-30-biomedical-retrieval-project-v4.1-patch.md
├── configs/
├── Dockerfile
├── docs/
├── environments/
├── LICENSE
├── README.md
├── resume-applied-scientist.pdf
├── resume_bio_v2.pdf
├── Resume_MLE.pdf
├── retrieval/
├── scripts/
├── serving/
└── tests/
```

Keep the existing v1 structure intact. Add:

```text
scientific-rag/
├── posttraining/                 # NEW
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── calibration/
│   ├── scaling/
│   └── utils/
├── configs/
│   └── posttraining/             # NEW
├── docs/
│   └── posttraining/             # NEW
└── tests/
    └── posttraining/             # NEW
```

Large generated artifacts stay outside Git.

**Repo hygiene (check in Week 0):**

- **Resume PDFs:** `resume-applied-scientist.pdf`, `resume_bio_v2.pdf`, and `Resume_MLE.pdf` sit at the repo root. If the repo is public on GitHub, they expose contact details and older resume versions. Move them out of the repo and purge them from git history, or confirm the repo is private.
- **Git inside Dropbox:** Dropbox syncing a `.git` folder can corrupt it under concurrent writes [COMMON, MED]. If you work from more than one machine, exclude `.git` from Dropbox sync, or push to the remote after every session.

---

## 3. Existing External Storage Layout

The previous project used:

```text
RAG_ROOT=/home/boweiye2/rag
```

Known paths:

| Purpose | Path |
|---|---|
| Heavy-artifact root | `/home/boweiye2/rag` |
| SciFact / derived data | `/home/boweiye2/rag/data/scifact` |
| Downloaded models | `/home/boweiye2/rag/models` |
| Hugging Face cache | `/home/boweiye2/rag/cache/huggingface` |
| Embeddings / FAISS indexes | `/home/boweiye2/rag/embeddings` |
| Checkpoints | `/home/boweiye2/rag/checkpoints` |
| Experiment outputs | `/home/boweiye2/rag/runs` |
| Logs | `/home/boweiye2/rag/logs` |
| Retrieval environment | `/home/boweiye2/rag/envs/retrieval` |
| Serving environment | `/home/boweiye2/rag/envs/serving` |

Existing setup is loaded with:

```bash
source scripts/env.sh
```

Recommended v2 additions:

```text
/home/boweiye2/rag/data/posttraining/
/home/boweiye2/rag/checkpoints/posttraining/
/home/boweiye2/rag/runs/posttraining/
/home/boweiye2/rag/logs/posttraining/
/home/boweiye2/rag/envs/posttraining/
```

---

## 4. Frozen v1 Components to Reuse

### Retrieval

Reuse the selected v1 retriever:

```text
Model: BAAI/bge-base-en-v1.5
Revision: a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
Query prefix: Represent this sentence for searching relevant passages:
Document format: title + abstract
Pooling: CLS
Embedding dtype: FP32
Normalization: L2
Search: FAISS IndexFlatIP
```

Main v2 studies keep retrieval frozen.

Do not retrain BGE in the primary post-training experiment.

### Verification baseline

Existing v1 verifier:

```text
Qwen/Qwen2.5-7B-Instruct
```

Existing pipeline:

```text
claim
    -> frozen BGE
    -> FAISS
    -> top-1 full abstract
    -> Qwen2.5-7B-Instruct
    -> SUPPORT / CONTRADICT / INSUFFICIENT
```

Historical v1 benchmark performance:

```text
Accuracy ≈ 0.700
Macro F1 ≈ 0.686
```

Historical v1 gold-document diagnostic:

```text
Accuracy ≈ 0.830
Macro F1 ≈ 0.800
```

---

## 5. Existing v1 Files to Inspect Before New Work

### Final held-out evaluation

```text
configs/final-test-6a-v1.json
docs/milestone6a-final-test-report.md
docs/milestone6a-final-test-results.json
```

External run:

```text
/home/boweiye2/rag/runs/milestone6a-test-v1
```

Important run artifacts include:

```text
pre-run-manifest.json
implementation-corrections.json
retrieval-rankings.json
retrieval-cosine-scores.json
retrieval-per-query.json
verification/
error-decomposition.json
selected-error-review.json
final-summary.json
source/
```

### Serving

Inspect:

```text
docs/milestone6b-serving-report.md
```

Existing serving run:

```text
/home/boweiye2/rag/runs/milestone6b-serving-v1
```

This is the source of truth for:

```text
vLLM settings
batch/concurrency settings
max model length
dtype
GPU-memory configuration
FastAPI settings
benchmark protocol
```

### Other relevant reports

```text
docs/scifact-mapping-report.md
docs/scifact-dev-baselines.md
docs/milestone4b-replication-review.md
docs/milestone4c-statistical-failure-analysis.md
docs/milestone5a-verification-dev-report.md
docs/milestone5b-reranking-dev-report.md
docs/milestone5c-passage-localization-dev-report.md
docs/final-project-summary.md
```

Use frozen configs and exact run snapshots rather than relying on memory when they disagree.

---

## 6. Previous Environment

The old project used separate retrieval and serving environments:

```text
/home/boweiye2/rag/envs/retrieval
/home/boweiye2/rag/envs/serving
```

Repository dependency records live under:

```text
environments/
```

One known milestone freeze is:

```text
environments/passage-localization-5c-pip-freeze.txt
```

Previously recorded serving stack:

```text
Python        3.11.17
PyTorch       2.7.x with CUDA 12.6 build
Transformers  4.53.2
vLLM          0.9.2
FastAPI       0.116.1
FAISS         1.11.x
```

These are reference values only. Before v2, recover the exact v1 dependency freeze and record the actual installed versions.

---

## 7. New Post-Training Environment

Do not modify the old retrieval or serving environments in place.

Recommended new environment:

```text
/home/boweiye2/rag/envs/posttraining
```

Recommended dependency freeze:

```text
environments/posttraining-v1-pip-freeze.txt
```

Expected packages:

```text
torch
transformers
datasets
accelerate
peft
trl
bitsandbytes       # only if QLoRA is used
numpy
pandas
scipy
scikit-learn
faiss
pytest
```

For distributed full fine-tuning:

```text
torch.distributed
FSDP
NCCL
```

DeepSpeed remains optional.

Serving should continue using the existing serving environment unless a dependency change is necessary and documented.

---

## 8. Proposed v2 Configuration Files

Use:

```text
configs/posttraining/
```

Recommended files:

```text
protocol-v1.json
baseline-b0-7b-zero-v1.json
baseline-b1-7b-fewshot-v1.json
baseline-b2-3b-zero-v1.json
baseline-b3-3b-fewshot-v1.json
data-audit-v1.json
train-m1-lora-gold-v1.json
train-m2-lora-mixed-v1.json
train-m3-full-ft-v1.json
fsdp-scaling-v1.json
calibration-v1.json
final-eval-v1.json
serving-tuned-3b-v1.json
```

Each major run should copy the exact config into its external run directory.

---

## 9. Proposed v2 Run Paths

```text
/home/boweiye2/rag/runs/posttraining/
├── protocol-check-v1/
├── baselines-v1/
├── data-audit-v1/
├── cv-selection-v1/
├── lora-gold-v1/          # model M1
├── lora-mixed-v1/         # model M2
├── benchmark-eval-v1/     # B0-B3, M1, M2 on the fixed 300-example benchmark
├── full-ft-v1/            # model M3
├── fsdp-scaling-v1/
├── benchmark-eval-m3-v1/  # M3 only, one pre-declared benchmark run
└── serving-v1/
```

Never overwrite a completed run. Increment the version instead.

Run folders are named by content only. `M0`-`M3` and `B0`-`B3` always refer to models, never to milestones or runs.

---

# 10. Milestone 0: Verify v1 Configuration and Protocol

## Required checks

1. Confirm the identity of:
   - TRAIN = 647
   - DEV = 162
   - fixed SciFact benchmark split = 300

   The 300-example split was already evaluated in v1, so for v2 it must not be described as a pristine unseen test set. Use the wording: **fixed SciFact benchmark split, previously evaluated in v1 and not used for v2 training or model selection**.

   Label-structure check: SciFact labels belong to claim-abstract pairs [KNOWN, MED; verify here]. Count TRAIN claims whose annotated evidence abstracts carry **different** labels (some SUPPORT, some CONTRADICT). Record the count in the protocol-check run. If any exist, `claim_gold_label` (Section 12) is `MIXED` for those claims: exclude them from Conditions B, C and D, and use them only in Condition A with per-pair labels.

2. Recover the exact v1 scoring rule for:
   - top-1 is not an annotated evidence document.

3. Confirm the frozen retrieval configuration:
   - BGE revision
   - query prefix
   - document formatting
   - pooling
   - normalization
   - index type
   - embeddings/index paths

4. Recover the exact baseline vLLM setup:
   - vLLM version
   - dtype
   - tensor parallel size
   - max model length
   - max sequences / concurrency
   - GPU memory utilization
   - decoding settings

5. Check the context budget for few-shot prompts:
   - record v1's `max_model_len`
   - measure the token length of the longest planned few-shot prompt (3-5 example abstracts + evidence abstract + claim)
   - if it exceeds the limit, choose shorter example abstracts or a fixed token budget; never let prompts silently truncate

6. Repo hygiene: resume PDFs and git-in-Dropbox (Section 2).

7. Record available hardware:
   - GPU model
   - number of GPUs
   - VRAM / GPU
   - driver
   - CUDA
   - NCCL
   - SLURM/node setup if applicable

## Exit condition

```text
B0 reproduced
v1 scoring known
splits and benchmark status documented
retrieval frozen
vLLM config frozen
compute availability known
few-shot prompts fit max_model_len
```

---

# 11. Milestone 1: Baselines

Run:

```text
B0 = Qwen2.5-7B zero-shot
B1 = Qwen2.5-7B few-shot
B2 = Qwen2.5-3B zero-shot
B3 = Qwen2.5-3B few-shot
```

All use the same:

```text
frozen BGE retrieval
top-1 evidence
scoring rule
output schema
decoding setup
```

Run path:

```text
/home/boweiye2/rag/runs/posttraining/baselines-v1/
```

Use the serving environment and the frozen vLLM configuration from Milestone 0 (batch size / max concurrent sequences, max model length, dtype, vLLM version). The same configuration is later used for M1-M3 (Section 14).

**Split policy for Milestone 1:**

- B1-B3 baseline development/evaluation is performed on **DEV only**.
- B0 may additionally be reproduced on the fixed 300-example benchmark during Week 0 solely to verify compatibility with v1, and if necessary re-scored once under the final-v4 Section 3.2 fallback rule.
- B1-B3 do **not** run on the 300-example benchmark until the frozen benchmark evaluation in Milestone 6.

Required table:

| Model | Prompt | Accuracy | Macro F1 | Invalid Rate |
|---|---|---:|---:|---:|
| B0 7B | zero-shot | | | |
| B1 7B | few-shot | | | |
| B2 3B | zero-shot | | | |
| B3 3B | few-shot | | | |

---

# 12. Milestone 2: Build and Audit SFT Data

Evidence-conditioned training target:

```json
{
  "decision": "SUPPORT | CONTRADICT | INSUFFICIENT",
  "rationale_sentences": [2, 5]
}
```

The decision answers: **given this supplied context, does the context SUPPORT, CONTRADICT, or provide INSUFFICIENT evidence for the claim?** This is not always identical to the original SciFact label for the claim.

Input:

```text
claim + supplied evidence
```

Only answer/output tokens contribute directly to the SFT loss. The full claim/evidence prompt remains available through causal attention.

### Condition A: Gold evidence

```text
claim + annotated evidence
-> gold decision + rationale indices
```

### Condition B: Retrieved non-gold top-1

Pool B:

```text
filter-development sample >= 50
-> define and freeze filter
-> fresh frozen-filter audit >= 50
```

### Condition C: rank 2-10 hard negatives

Pool C uses its own:

```text
filter-development sample
filter
fresh audit
stop rule
```

Do not merge Pool B and Pool C estimates.

Audit wording:

- The first sample of each pool is the **filter-development sample**: it is used to design the filter, and its rate is reported only as the pre-filter rate.
- The second sample is the **audit of the frozen filter**, and its rate is the one compared against the 15% stop threshold.
- If you re-read your own items, call it **intra-rater consistency**. If a second person labels items independently, call it **inter-rater agreement**.

### Condition D: Random negatives

```text
claim + unrelated abstract
-> INSUFFICIENT
```

Recommended data paths:

```text
/home/boweiye2/rag/data/posttraining/
├── train_gold.jsonl
├── train_retrieved.jsonl
├── train_hard_negative.jsonl
├── train_random_negative.jsonl
├── train_mixed.jsonl
└── audits/
    ├── pool-b-filter-development.csv
    ├── pool-b-frozen-filter-audit.csv
    ├── pool-c-filter-development.csv
    └── pool-c-frozen-filter-audit.csv
```

Every example should keep provenance fields and preserve the original SciFact labels separately from the evidence-conditioned training label:

```text
claim_id
doc_id
condition
source_split
claim
evidence
claim_gold_label
pair_annotation
context_label
rationale_sentences
annotation_basis
```

The distinction is mandatory:

```text
claim_gold_label
    = the claim's SciFact label on its annotated evidence abstract(s):
      SUPPORT / CONTRADICT, NOT_ENOUGH_INFO if the claim has no evidence,
      MIXED if its evidence abstracts disagree (see Section 10 check)

pair_annotation
    = the SciFact label for THIS exact claim-document pair:
      SUPPORT / CONTRADICT if the pair is annotated, null otherwise

context_label
    = what the supplied evidence context supports
      (the SFT training target):
      SUPPORT / CONTRADICT / INSUFFICIENT
```

Consistency rules:

- If `pair_annotation` is not null, `context_label` must equal `pair_annotation` (Condition A).
- If `pair_annotation` is null, `context_label` is INSUFFICIENT and the example must have passed its pool's frozen filter (Conditions B and C) or be a random negative (Condition D).
- Claims with `claim_gold_label = MIXED` appear only in Condition A.
- A unit test in `tests/posttraining/` enforces these rules on every generated JSONL file.

Examples:

```json
{
  "claim_gold_label": "SUPPORT",
  "pair_annotation": "SUPPORT",
  "context_label": "SUPPORT",
  "condition": "gold"
}
```

```json
{
  "claim_gold_label": "SUPPORT",
  "pair_annotation": null,
  "context_label": "INSUFFICIENT",
  "condition": "retrieved_non_gold"
}
```

Never overwrite `claim_gold_label` or `pair_annotation` when constructing evidence-conditioned negatives.

Exit condition:

```text
no DEV/held-out leakage
A/B duplicates removed
Pool B audited
Pool C audited
effective mixture recorded
```

---

# 13. Milestone 3: LoRA SFT

This is the primary modeling milestone.

### M1

```text
Qwen2.5-3B-Instruct
+ LoRA
+ gold evidence only
```

Comparison:

```text
M1 vs B2
```

Question:

> Does SFT help at fixed model size?

### M2

```text
Qwen2.5-3B-Instruct
+ LoRA
+ gold + filtered retrieved + filtered hard negatives + random negatives
```

Comparisons:

```text
M2 vs M1
M2 vs B3
M2 vs B0
M2 vs B1
```

Questions:

```text
Does mixed-evidence training help?
Does SFT beat few-shot prompting?
Can tuned 3B compete with 7B?
```

Config paths:

```text
configs/posttraining/train-m1-lora-gold-v1.json
configs/posttraining/train-m2-lora-mixed-v1.json
```

Checkpoint paths:

```text
/home/boweiye2/rag/checkpoints/posttraining/m1/
/home/boweiye2/rag/checkpoints/posttraining/m2/
```

---

# 14. Milestone 4: Model Selection and Evaluation

Use:

```text
grouped 5-fold CV on TRAIN
```

All examples derived from one claim stay in the same fold.

Use one fixed seed in CV.

Then evaluate 2-3 finalists on DEV with 3 training seeds.

**Finalist selection rule (declared now):**

```text
choose the finalist with the best 3-seed mean DEV Macro F1
-> its seed 0 is the inferential checkpoint
```

- Ties (within 0.005 Macro F1 [GUESS]) go to the simpler configuration.
- Seed 0's own DEV score never decides the finalist.
- Seeds 1-2 are stability checks only.

**Evaluation path for tuned models:** merge the LoRA weights into the base model, then evaluate through the same serving environment and frozen vLLM configuration as B0-B3. Train in the post-training environment, evaluate in the serving environment. Record which stack produced calibration scores (vLLM log-probs or Hugging Face forward pass); small numeric differences between the two are expected and covered by the agreement check.

**Primary metrics on E1 (retrieved top-1):**

```text
Accuracy
Macro F1
```

**Secondary metrics:**

```text
class-level F1
predicted vs true label distribution, per-class recall (over-abstention check)
JSON validity, decision validity
rationale-index validity, rationale precision/recall/F1
ECE, Brier, NLL (restricted-choice scoring)
```

Evaluation conditions:

```text
E1 = retrieved top-1 evidence                    # primary
E2 = annotated evidence                          # secondary
E3 = constructed hard negatives                  # secondary, constructed
E4 = constructed insufficient evidence contexts  # secondary, constructed
```

All main v2 benchmark claims rest on E1.

Statistics:

```text
Accuracy:
McNemar + Holm across the five primary comparisons (tests 1-5)
M3 vs M2 (test 6): secondary, reported separately

Macro F1:
paired bootstrap 95% CI (unadjusted)
```

Seed consistency (definition):

```text
at least 2 of 3 seeds, including seed 0, show the same sign of effect,
and the 3-seed mean difference has the same sign
```

---

# 15. Milestone 5: Full Fine-Tuning and FSDP

M3:

```text
Qwen2.5-3B-Instruct
full-parameter fine-tuning
```

Comparison:

```text
M3 vs M2
```

Question:

> Does full fine-tuning improve enough over LoRA to justify the training cost?

M3 is secondary and must not block completion of the project.

**Benchmark-evaluation policy for M3 (declared now):**

- The main v2 benchmark run (B0-B3, M1, M2) happens first and is never repeated.
- M3 is frozen on its own, then evaluated on the same fixed SciFact benchmark split **exactly once for M3** in `benchmark-eval-m3-v1/`.
- Test 6 (M3 vs M2) uses the M2 predictions already saved from the main benchmark run.
- Nothing about M1/M2 or the main results changes because of the M3 run.
- If M3 is abandoned, record that it was abandoned and why; there is no M3 benchmark run.
- Because this 300-example split was already evaluated in v1, describe these results as a **fixed benchmark comparison**, not as evaluation on a never-before-seen test set.

---

## 15.1 Why full training may require multiple GPUs

Approximate 3B BF16 inference weights:

```text
~6 GB
```

Approximate full AdamW training state:

| Component | Approx. memory |
|---|---:|
| BF16 parameters | 6 GB |
| gradients | 6 GB |
| FP32 master weights | 12 GB |
| Adam first moment | 12 GB |
| Adam second moment | 12 GB |
| subtotal | ~48 GB |

This excludes:

```text
activations
temporary tensors
attention intermediates
CUDA overhead
fragmentation
```

Therefore:

```text
3B inference on 48 GB -> easy
3B LoRA training on 48 GB -> feasible
3B full AdamW training on 48 GB -> may OOM / be impractical
```

---

## 15.2 FSDP role

```text
LoRA:
Which parameters are updated?

FSDP:
How full-training state is sharded across GPUs.
```

Scaling configurations where feasible:

```text
1 GPU
2 GPUs
4 GPUs
```

If 1 x 48 GB fails:

```text
record OOM
use 2 GPUs as baseline
compare 2 vs 4
```

No CPU offload in the scaling series.

Hold fixed:

```text
model
precision
sequence length
data
optimizer
learning-rate schedule
global batch size
per-GPU micro-batch
optimizer steps
gradient checkpointing
```

Change only:

```text
GPU count
gradient accumulation
```

Example:

| GPUs | Micro-batch / GPU | Accumulation | Global Batch |
|---:|---:|---:|---:|
| 1 | 2 | 16 | 32 |
| 2 | 2 | 8 | 32 |
| 4 | 2 | 4 | 32 |

Measure:

```text
fit/OOM
tokens/sec
samples/sec
peak memory/GPU
wall-clock time
communication overhead if available
scaling efficiency
```

More GPUs are expected to improve fit and speed, not inherently improve model quality.

---

# 16. Milestone 6: Frozen Benchmark Evaluation

Before the v2 benchmark run, freeze:

```text
model
checkpoint rule
retrieval configuration
prompt
evidence depth
decoding
parser
label mapping
scoring rule
rationale scorer
calibration scorer
statistics plan
software environment
```

Config:

```text
configs/posttraining/final-eval-v1.json
```

Run:

```text
/home/boweiye2/rag/runs/posttraining/benchmark-eval-v1/       # B0-B3, M1, M2
/home/boweiye2/rag/runs/posttraining/benchmark-eval-m3-v1/    # M3 only, after its own freeze
```

Seed policy:

```text
finalist chosen by 3-seed DEV mean
seed 0 of that finalist = inferential checkpoint
seed 1-2 = stability only
```

No post-benchmark tuning based on the 300-example results.

---

# 17. Milestone 7: Serving and Integration

Integrate the final selected model into:

```text
claim
    -> frozen BGE
    -> FAISS
    -> top-1 abstract
    -> tuned Qwen2.5-3B
    -> decision + rationale IDs
```

Likely selected model:

```text
M2
```

unless M3 provides a clearly justified gain.

Compare against v1 7B serving.

Metrics:

```text
P50 latency
P95 latency
requests/sec
GPU memory
retrieval latency
generation latency
failure rate
invalid-output rate
```

Existing v1 reference:

```text
GPU: NVIDIA L40S
Concurrency 8: 4.34 req/s
P50: ~1.65 s
P95: ~2.43 s
Retrieval: ~13-22 ms
Peak GPU memory: ~34.9 GiB
```

Serving config:

```text
configs/posttraining/serving-tuned-3b-v1.json
```

Serving run:

```text
/home/boweiye2/rag/runs/posttraining/serving-v1/
```

---

# 18. Reproducibility Requirements

For each major run, save:

```text
git commit
config
config hash
model revision
tokenizer revision
hardware
CUDA
driver
Python
PyTorch
Transformers
PEFT
TRL
vLLM, if applicable
FAISS
NCCL, if distributed
pip freeze
metrics
```

Recommended files:

```text
manifest.json
config.json
environment.txt
pip-freeze.txt
git-commit.txt
hardware.json
metrics.json
```

---

# 19. Week 0 Commands

Start with:

```bash
cd /home/boweiye2/Dropbox/Non-coding-variant/scientific-rag
source scripts/env.sh
```

Inspect frozen v1 material:

```bash
cat configs/final-test-6a-v1.json
cat docs/milestone6a-final-test-report.md
cat docs/milestone6b-serving-report.md
```

Repository:

```bash
git status
git rev-parse HEAD
git remote -v                      # is the remote public?
git ls-files '*.pdf'               # resume PDFs tracked?
```

Tests:

```bash
python -m pytest -q
```

Environment:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
python -c "import faiss; print(faiss.__version__)"
```

Serving environment:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
```

Few-shot context budget (after the few-shot prompts are drafted):

```bash
python -c "from transformers import AutoTokenizer as T; t=T.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); print(len(t(open('fewshot_prompt_longest.txt').read())['input_ids']))"
```

Hardware:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

Do not upgrade or reinstall packages until the old environment and freeze have been recorded.

---

# 20. Immediate Execution Order

1. Confirm the origin of the 300-example SciFact benchmark split and document that it was previously evaluated in v1.
2. Recover v1's exact non-annotated top-1 scoring rule.
3. Reproduce B0.
4. Record exact software, config, and hardware state.
5. Confirm multi-GPU access.
6. Create the separate post-training environment.
7. Create `configs/posttraining/`, `posttraining/`, `docs/posttraining/`, and `tests/posttraining/`.
8. Check few-shot prompt length against `max_model_len`; clean up repo hygiene.
9. Run B1-B3.
10. Build and audit Pool B and Pool C (filter-development sample, then frozen-filter audit).
11. Grouped CV for M1/M2; DEV finalists with 3 seeds; select by 3-seed mean.
12. Freeze and run the main v2 benchmark evaluation (B0-B3, M1, M2), once.
13. Run M3 and the FSDP scaling study only after the core LoRA study is complete.
14. Freeze M3; run its single pre-declared benchmark evaluation; compute test 6.
15. Integrate and benchmark the selected model in serving.

---

# 21. Stop Conditions

Do not start SFT if:

```text
B0 cannot be reproduced
v1 scoring is unresolved AND the final-v4 Section 3.2 fallback rule
    has not yet been adopted with B0 re-scored under it
split identity or prior v1 benchmark usage is unresolved
retrieval config is not frozen
few-shot prompts exceed max_model_len without a fixed budget
```

An unclear v1 rule is not a reason to stall: adopt the fallback rule, re-score B0 under it, keep the historical 0.700 under the old rule, and proceed.

Do not use noisy Pool B/C examples as supervised INSUFFICIENT labels if the independent frozen-filter audit fails the protocol.

Do not let the full-fine-tuning/FSDP study block completion of the primary LoRA result.

---

# 22. Final Project Story

```text
v1:
Built retrieval + zero-shot verification.

Observation:
Better retrieval metrics did not guarantee better verification,
and Qwen still made errors when correct evidence was supplied.

v2:
Post-train the verifier.

Study 1:
Does SFT improve 3B at fixed model size?

Study 2:
Does realistic mixed-evidence training improve robustness?

Study 3:
Can tuned 3B match or beat 7B zero/few-shot?

Study 4:
Does full fine-tuning justify its additional compute?

Systems:
How does full-parameter training scale across GPUs?

Evaluation:
How much does post-training improve performance on the fixed SciFact benchmark used previously in v1?

Optional external extension:
If desired, add a genuinely new compatible dataset later for out-of-distribution generalization.

Deployment:
Does tuned 3B improve the quality / latency / memory tradeoff?
```

Use this implementation plan unless Week 0 reveals a concrete inconsistency in the old split, scoring rule, configuration, or environment.

---

# 23. Protocol Items Carried from final-v4 (frozen analysis plan)

These go into `configs/posttraining/protocol-v1.json` and are frozen before any SFT run.

| Item | Rule |
|---|---|
| Primary endpoint | Fixed SciFact benchmark split, E1 (retrieved top-1 evidence), scored by v1's rule or the final-v4 3.2 fallback; previously evaluated in v1 and not used for v2 training/model selection |
| Primary metrics | Accuracy, Macro F1 |
| Pre-registered tests | 1 M1-B2, 2 M2-B3, 3 M2-M1, 4 M2-B0, 5 M2-B1 (Holm family); 6 M3-M2 (secondary) |
| Inferential checkpoint | Seed 0 of the finalist chosen by 3-seed DEV mean |
| Seed consistency | ≥2 of 3 seeds including seed 0 same sign, and the 3-seed mean same sign |
| Mixture options | Mix-1 45/20/20/15, Mix-2 60/15/15/10, Mix-3 60/25*/0/15 (final-v4 Section 6); report effective counts |
| Audit stop threshold | Frozen-filter audit rate >15% → drop that pool (project heuristic) |
| Audit samples | ≥50 filter-development + ≥50 frozen-filter audit, per pool, Pools B and C separate |
| Calibration | Restricted-choice label scoring in the `{"decision": "` prefix; ECE, Brier, NLL; report agreement with generated decisions |
| Rationale scope | Only annotated evidence with a SUPPORT/CONTRADICT label; max over gold rationale sets; report the exclusion count |
| Over-abstention | Report predicted vs true label distribution and per-class recall on E1 |
| Validity | JSON/schema validity and rationale-index validity reported separately |
| Success criteria | As final-v4 Section 25 (strong / moderate / negative-but-informative), with the seed-consistency rule above |
| Few-shot rules | Same fixed 3-5 TRAIN examples for B1 and B3, all three labels, chosen before evaluation, within the token budget |
| Label semantics | Keep `claim_gold_label` and `pair_annotation` separate from the evidence-conditioned `context_label`; SFT trains on `context_label`; consistency rules enforced by unit test; MIXED claims only in Condition A |
| Condition naming | A-D are reserved for training-data construction only; E1-E4 are reserved for evaluation conditions |
| Benchmark disclosure | 300-example split = fixed benchmark previously evaluated in v1 and used to motivate v2's design; report seed-0 DEV results alongside; only B0 reproduction/re-scoring touches it before the freeze |
| LoRA defaults | Rank 16, alpha 32, dropout 0.05, learning rate ≈2e-4, 2-3 epochs; only epochs, mixture, and LoRA vs full fine-tuning are selected |
