# Biomedical Evidence Verification v2
## Task-Specific LLM Post-Training for Evidence-Grounded Scientific Verification

**Date:** 2026-10-08  
**Version:** final-v4 (final-v3 + two fixes and two tweaks)

---

# Change Log: final-v4

Four edits on top of final-v3. Everything else is unchanged.

| # | Fix | Sections | Why |
|---|---|---|---|
| 1 | The tested checkpoint is **pre-declared as seed 0**, never chosen on DEV | Change log, 9.3, 15.1, 15.3 | final-v3 said both "DEV-selected" and "pre-declared". Picking the best of 3 seeds on 162 DEV claims adds a small lucky-pick bias, so the rule must be fixed before training. |
| 2 | Non-annotated top-1 documents get **their own audit and their own filter rule** | 5.2, 5.3, 6 | They rank *above* the gold document by definition, so a "close to the gold score" filter kept the documents most likely to be truly relevant and labeled them INSUFFICIENT. Also covers claims whose gold document isn't retrieved at all. |
| 3 | Holm correction over **primary tests 1–5 only**; test 6 (M3 vs M2) reported separately | 15 | M3 is a secondary question, so including it in the Holm family lowers power for the primary tests. |
| 4 | Fixed vLLM batch configuration for baseline runs | 9.3 | vLLM greedy decoding isn't always exactly reproducible across batch sizes [COMMON, MED], so "run once" baselines need a fixed setup to be reproducible. |

**Alternative to fix 1:** use a majority vote of the 3 seeds as the single frozen predictor (ties broken by the highest mean restricted-choice probability). It uses all seeds without any picking. If you prefer it, change Sections 9.3, 15.1 and 15.3 together.

---

# Change Log: final-v3

This version keeps the final-v2 design and makes four final corrections:

1. **Separate multiplicity handling for Accuracy and Macro F1**
   - Accuracy: McNemar tests with Holm correction across the pre-registered comparisons.
   - Macro F1: paired bootstrap 95% confidence intervals on the difference.
   - Do not describe Macro F1 as "Holm-corrected" unless bootstrap p-values are explicitly computed and corrected.

2. **Do not treat three training seeds as three separate hypothesis tests**
   - Use one pre-declared frozen checkpoint (seed 0, see final-v4 fix 1) for inferential paired tests.
   - Use the remaining seeds to quantify training stability.
   - This avoids silently turning 6 pre-registered comparisons into 18 hypothesis tests.

3. **Describe calibration scores as restricted-choice label probabilities**
   - The normalized probabilities are conditional on the output belonging to the three allowed labels.
   - Keep the agreement check between restricted-choice scoring and actual structured generation.

4. **Separate structural output validity from rationale-citation validity**
   - Malformed JSON and invalid sentence indices are different failure modes.
   - Report them separately.

Everything else is preserved.

---

# Protocol Clarification: Evaluation-Condition Naming

**Evaluation-condition naming patch:** Section 5 training conditions retain labels A-D. Evaluation conditions use E1-E4 throughout: E1 retrieved top-1, E2 annotated evidence, E3 constructed hard negatives, E4 constructed insufficient evidence. This resolves the earlier A-D naming collision without changing any experimental design, metric, endpoint, or statistical test.

**Benchmark-terminology patch:** Wherever this document says "held-out split," read "fixed SciFact benchmark split, previously evaluated in v1 and not used for v2 training or model selection." See the implementation document's Benchmark status section and benchmark-reuse disclosure.

---

# 0. Design Summary

```text
B0: Qwen2.5-7B zero-shot
B1: Qwen2.5-7B few-shot
B2: Qwen2.5-3B zero-shot
B3: Qwen2.5-3B few-shot

M1: Qwen2.5-3B + LoRA SFT, gold evidence
M2: Qwen2.5-3B + LoRA SFT, mixed evidence
M3: Qwen2.5-3B full fine-tune + FSDP
M4: optional GRPO extension
```

Main research questions:

1. Does SFT improve evidence-grounded verification beyond zero-shot and few-shot prompting?
2. Does mixed-evidence training improve robustness to imperfect retrieval?
3. Can a tuned 3B model match or beat a 7B zero-shot/few-shot model?
4. Does full fine-tuning provide enough gain over LoRA to justify the extra compute?
5. How does full fine-tuning scale across multiple GPUs?

The main modeling result must not depend on M3 succeeding.

---

# 1. Motivation

Current pipeline:

```text
claim
  -> BGE retrieval
  -> FAISS
  -> top-1 abstract
  -> Qwen2.5-7B-Instruct
  -> SUPPORT / CONTRADICT / INSUFFICIENT
```

Existing held-out result:

```text
Accuracy ≈ 0.700
Macro F1 ≈ 0.686
```

Two findings motivate the extension:

- Better document ranking did not necessarily improve downstream verification.
- Reasoning errors remained even when annotated evidence was supplied.

Therefore, retrieval is not the only bottleneck.

Core question:

> Can task-specific post-training make the verifier use evidence more effectively, especially when the evidence is imperfect?

---

# 2. Research Questions and Direct Comparisons

| RQ | Comparison | What It Isolates |
|---|---|---|
| RQ1a | M1 vs B2 | SFT effect at fixed 3B size |
| RQ1b | M2 vs B3 | SFT vs prompting at fixed 3B size |
| RQ2 | M2 vs M1 | Mixed-evidence training vs gold-only training |
| RQ3a | M2 vs B0 | Tuned 3B vs zero-shot 7B |
| RQ3b | M2 vs B1 | Tuned 3B vs few-shot 7B |
| RQ4 | M3 vs M2 | Full fine-tuning vs LoRA |
| RQ5 | M3 on 1 / 2 / 4 GPUs | Strong-scaling behavior |

---

# 3. Data, Splits, and Scoring Rule

## 3.1 Splits

- Corpus: 5,183 documents
- TRAIN: 647 claims
- DEV: 162 claims
- Held-out split: 300 claims

Before training begins, confirm exactly where the 300-claim split comes from.

If it is SciFact's official development split, describe it everywhere as:

> held-out SciFact split (official development split)

Do not call it the official test set unless that is verified.

---

## 3.2 Gold Label on Retrieved Evidence

The deployed primary endpoint uses BGE top-1 evidence, so the gold scoring rule must be explicit.

### Preferred rule

1. Reuse the exact v1 evaluation rule if it is already documented and reproducible.
2. Copy that rule into the README.
3. Reproduce B0 under that rule before any v2 comparisons.

If the old rule is unclear, use:

| Situation | Gold Label |
|---|---|
| Top-1 is an annotated evidence document | SciFact label for that document |
| Top-1 is not annotated and claim is SUPPORT/CONTRADICT | INSUFFICIENT, evidence considered unverified |
| Claim is NOT ENOUGH INFO | INSUFFICIENT |

If this differs from v1:

- retain historical B0 numbers under the old rule
- re-score B0 under the new rule
- use the new rule consistently for all v2 comparisons

### Known limitation

SciFact evidence annotations are not necessarily exhaustive.

A non-annotated retrieved document may genuinely support or contradict the claim.

Therefore:

- estimate this error rate with the hard-negative audit
- report it as possible endpoint label noise
- do not hide it

---

# 4. Training Target

Use:

```json
{
  "decision": "SUPPORT | CONTRADICT | INSUFFICIENT",
  "rationale_sentences": [2, 5]
}
```

Rationale:

- decision labels are directly supervised
- rationale sentence indices exist in SciFact
- no teacher LLM is needed
- no free-text explanation confound is introduced
- rationale quality becomes directly measurable

For INSUFFICIENT examples:

```json
{
  "decision": "INSUFFICIENT",
  "rationale_sentences": []
}
```

Free-text explanations, if desired for the UI, should be generated after the decision and kept outside the main evaluated model target.

---

# 5. Training Data Construction

## 5.1 Condition A: Gold Evidence

Input:

```text
claim + annotated evidence abstract
```

Target:

```text
gold decision + rationale indices
```

Purpose:

Teach clean evidence-grounded reasoning.

---

## 5.2 Condition B: Retrieved Evidence

Input:

```text
claim + BGE top-1 abstract
```

Rules:

| Case | Action |
|---|---|
| Top-1 is annotated gold evidence | Drop from B, since it duplicates A |
| Top-1 is non-annotated | Goes to its **own audit pool (Pool B)**, separate from the rank 2-10 pool. See 5.3 |
| Pool B passes its own stop rule | Non-annotated top-1 examples that pass the Pool B filter are labeled INSUFFICIENT |
| Pool B fails its stop rule | Drop all non-annotated top-1 examples (default fallback: Mix-3 without B) |

After de-duplication, Condition B represents real retrieval failures rather than duplicated gold examples.

---

## 5.3 Condition C: Audited Hard Negatives

Candidate pool:

```text
BGE rank 2-10 documents that are not annotated evidence
```

plus the non-annotated top-1 candidates from Condition B.

### Two audit pools (final-v4)

| Pool | Contents | Why separate |
|---|---|---|
| Pool B | Non-annotated BGE top-1 documents | They outrank the gold document, so they are probably the most likely to be truly relevant. Their mislabel rate is expected to be higher. [INFERRED, MED] |
| Pool C | Non-annotated BGE rank 2-10 documents | Standard hard negatives |

Run the audit procedure below **separately for each pool**, with its own samples, its own filter, and its own stop rule. Never pool the two estimates.

### Filter rules must not assume candidates rank below gold

A filter of the form "drop if the BGE score is within δ of the gold document's score" only makes sense for candidates ranked below the gold document. It fails in two cases:

- **Pool B candidates rank above gold.** The rule would keep exactly those that most outscore the gold document.
- **The gold document isn't retrieved in the top 10.** There is no gold score to compare against.

Acceptable filter types (choose and write down per pool before the re-audit):

- **Absolute score cut:** drop candidates whose claim-to-document similarity exceeds a threshold τ, chosen on the initial audit sample. This works whether or not gold was retrieved.
- **Lexical/entity overlap cut:** drop candidates sharing the claim's key entities (e.g., the drug and outcome terms) above a threshold.
- **No filter:** for Pool B, simply drop all of them. This is the recommended default if the initial Pool B audit is noisy.

### Audit procedure (per pool)

1. Randomly sample at least 50 candidates from TRAIN claims.
2. Manually judge whether each candidate actually supports or contradicts its claim.
3. Estimate the pre-filter mislabel rate with a 95% confidence interval.
4. Define the filtering rule using TRAIN only.
5. Write the filtering rule down before the re-audit.
6. Draw a fresh post-filter sample of at least 50.
7. Re-estimate the post-filter mislabel rate.
8. If residual label noise remains clearly high, for example above 15%, drop that pool: Pool C → drop Condition C; Pool B → drop non-annotated Condition B examples.

The 15% value is a project heuristic, not a benchmark standard.

Optional:

- a second manual read of 20 items for consistency checking
- report simple agreement

---

## 5.4 Condition D: Random Negatives

Input:

```text
claim + unrelated scientific abstract
```

Target:

```text
INSUFFICIENT
```

Purpose:

Provide a clean insufficient-evidence contrast.

---

# 6. Evidence Mixtures

Use only a small predefined set.

| Option | Gold A | Retrieved B | Hard Negative C | Random D |
|---|---:|---:|---:|---:|
| Mix-1 | 45% | 20% | 20% | 15% |
| Mix-2 | 60% | 15% | 15% | 10% |
| Mix-3, if C dropped | 60% | 25%* | 0% | 15% |

\*Use only Condition B examples that pass the Pool B filter and stop rule. If Pool B fails, drop them rather than forcing the target proportions; the mixture then becomes gold + random negatives only, and that is reported.

Selection:

```text
grouped cross-validation on TRAIN
```

Always report the effective post-filter training counts, not only intended percentages.

---

# 7. Models and Baselines

| ID | Model | Training | Purpose |
|---|---|---|---|
| B0 | Qwen2.5-7B-Instruct | zero-shot | historical baseline |
| B1 | Qwen2.5-7B-Instruct | few-shot | 7B prompting control |
| B2 | Qwen2.5-3B-Instruct | zero-shot | size control |
| B3 | Qwen2.5-3B-Instruct | few-shot | 3B prompting control |
| M1 | Qwen2.5-3B-Instruct | LoRA SFT, gold only | basic SFT |
| M2 | Qwen2.5-3B-Instruct | LoRA SFT, mixed evidence | evidence-robust SFT |
| M3 | Qwen2.5-3B-Instruct | full fine-tune + FSDP | secondary science + systems |
| M4 | best of M2/M3 | optional GRPO | exploratory |

Few-shot rules:

- same fixed 3-5 TRAIN examples for B1 and B3
- include all three labels
- same output schema as SFT
- examples chosen before evaluation

---

# 8. LoRA Configuration

Use a conservative fixed configuration.

Example:

```text
rank = 16
alpha = 32
dropout = 0.05
learning rate ≈ 2e-4
epochs = 2-3
```

Only select:

- epoch count
- evidence mixture
- possibly LoRA vs full fine-tuning

Avoid a broad hyperparameter sweep.

---

# 9. Model Selection Protocol

## 9.1 Grouped 5-Fold Cross-Validation on TRAIN

Rules:

- all examples derived from one claim stay in the same fold
- use one fixed seed for cross-validation
- few-shot exemplar claims are excluded from the relevant validation fold
- do not tune repeatedly on DEV

Purpose:

Prevent claim-level leakage and DEV overfitting.

---

## 9.2 DEV Confirmation

Select only 2-3 finalists from CV.

For each finalist:

```text
3 independent training seeds
```

Report:

```text
mean ± SD
```

If seed variation is comparable to the model differences, add more seeds before choosing.

---

## 9.3 Freeze Before Held-Out Evaluation

Freeze:

- model family
- checkpoint selection rule
- prompt
- evidence depth
- decoding
- parser
- label mapping
- scoring rule
- rationale evaluation
- calibration procedure
- statistical analysis plan
- evaluation code

Then evaluate the held-out split.

### Seed rule

All three frozen training seeds may be evaluated on the held-out split to quantify stability.

However:

- no seed is selected after seeing held-out results
- the held-out split is not used to choose a preferred seed
- inferential hypothesis tests use one pre-declared frozen checkpoint: **seed 0**, declared now, before any training
- the DEV results of seeds 1-2 never change which checkpoint is tested
- the other seeds are stability analyses, not separate hypothesis tests

Baselines B0-B3 use greedy decoding and run once, with a **fixed vLLM configuration** (batch size / max concurrent sequences, max model length, dtype, vLLM version). Record it in the README. vLLM greedy output can differ slightly across batch sizes [COMMON, MED], so reproduce B0 under this exact setup in Week 0.

---

# 10. Evaluation Conditions

To avoid collision with the training-data Conditions A-D in Section 5, evaluation conditions use a separate E-series:

| Evaluation condition | Status |
|---|---|
| E1. Retrieved top-1 evidence | **Primary endpoint** |
| E2. Annotated evidence | Secondary |
| E3. Constructed hard negatives | Secondary, constructed |
| E4. Constructed insufficient evidence | Secondary, constructed |

All main claims rest on E1.

**Naming convention:** A-D are reserved for training-data construction. E1-E4 are reserved for evaluation.

---

# 11. Over-Abstention Check

Mixed-evidence training may make the model over-predict:

```text
INSUFFICIENT
```

Therefore report:

- predicted label distribution
- true label distribution
- SUPPORT recall
- CONTRADICT recall
- INSUFFICIENT recall

A headline gain should not be interpreted as improvement if it is driven primarily by collapse toward INSUFFICIENT.

---

# 12. Metrics

## Primary

```text
Accuracy
Macro F1
```

on E1.

## Per Class

```text
F1_SUPPORT
F1_CONTRADICT
F1_INSUFFICIENT
```

## Structural Output Quality

Report separately:

```text
JSON validity
decision-field validity
schema validity
```

## Rationale Quality

Report separately:

```text
rationale-index validity
rationale precision
rationale recall
rationale F1
```

An invalid rationale index does not automatically imply malformed JSON.

Keep structural generation failures and citation failures separate.

## Calibration

```text
ECE
Brier Score
NLL
```

using the restricted-choice label-scoring path.

## Efficiency

- latency P50
- latency P95
- throughput
- peak GPU memory
- output tokens
- optional serving cost proxy

Use the existing vLLM/FastAPI benchmark harness where possible.

---

# 13. Rationale Evaluation

Only score rationale quality when:

- the evidence shown is an annotated evidence document
- the true decision is SUPPORT or CONTRADICT
- a gold rationale set exists

Exclude other examples and report the exclusion count.

Let:

```text
R_hat = predicted rationale sentence set
G = set of valid annotated rationale sets
```

Use:

```text
Rationale F1
=
max over G_i in G of F1(R_hat, G_i)
```

Reason:

SciFact may provide multiple valid sufficient rationale sets.

Also report:

```text
rationale-index validity rate
```

defined as the fraction of generated rationale indices that correspond to real sentences in the supplied abstract.

---

# 14. Calibration

Calibration should use a label-scoring path aligned with the same structured-generation context.

For each label:

```text
SUPPORT
CONTRADICT
INSUFFICIENT
```

score:

```text
<prompt> + '{"decision": "' + label
```

Compute the total sequence log-likelihood:

```text
l_k
```

Then normalize across only the three valid labels:

```text
p_k = exp(l_k) / Σ_j exp(l_j)
```

These are:

> normalized restricted-choice label probabilities

They are conditional on the answer belonging to the three allowed decision strings.

Use them for:

```text
ECE
Brier Score
NLL
```

Do not describe them as unrestricted model probabilities.

### Agreement check

Report:

```text
argmax restricted-choice label
vs.
actual structured-generation decision
```

If agreement is materially below approximately 95%, flag that the calibration path and generation path differ.

Structured generation remains the official prediction path for task metrics.

---

# 15. Statistical Comparisons

Pre-register six comparisons in two families:

| # | Comparison | Family |
|---|---|---|
| 1 | M1 vs B2 | Primary |
| 2 | M2 vs B3 | Primary |
| 3 | M2 vs M1 | Primary |
| 4 | M2 vs B0 | Primary |
| 5 | M2 vs B1 | Primary |
| 6 | M3 vs M2 | Secondary |

---

## 15.1 Accuracy

For the pre-declared frozen checkpoint (seed 0 for M1-M3):

```text
McNemar test
```

on paired claim-level predictions.

Apply:

```text
Holm correction
```

across the **five primary** Accuracy tests (1-5).

Test 6 (M3 vs M2) is the secondary family, reported on its own with its raw p-value, and labeled secondary.

Report:

- raw p-value
- Holm-adjusted p-value
- paired accuracy difference

---

## 15.2 Macro F1

Use:

```text
paired bootstrap 95% CI
```

for:

```text
Delta Macro F1
```

with, for example:

```text
10,000 claim-level bootstrap resamples
```

Do not call this "Holm-corrected" unless bootstrap p-values are explicitly computed and included in the same multiplicity correction.

Preferred interpretation:

```text
Accuracy -> McNemar + Holm
Macro F1 -> paired bootstrap CI
```

---

## 15.3 Seed Stability

For M1-M3:

```text
seed 1 metric
seed 2 metric
seed 3 metric
mean ± SD
```

Seeds are primarily a stability analysis.

They are not treated as three additional independent hypothesis tests.

Inferential paired tests use one pre-declared frozen checkpoint (seed 0).

The remaining seeds should agree qualitatively in direction with the main result.

---

# 16. Full Fine-Tuning Study

M3 is secondary.

Question:

> Does updating all 3B parameters improve enough over LoRA to justify the extra cost?

All outcomes are useful:

### M3 > M2

Full fine-tuning adds quality.

### M3 ≈ M2

Parameter-efficient tuning is sufficient.

### M3 < M2

Small-data overfitting or optimization difficulty may dominate.

The project is complete even if M3 does not help.

---

# 17. Distributed Training Study

Use M3 as the workload.

Configurations:

| Configuration | Status |
|---|---|
| 1 GPU | only if it fits without CPU offload |
| 2 GPUs + FSDP | fallback baseline |
| 4 GPUs + FSDP | target |

If 1 GPU is out of memory:

- report the OOM outcome
- use 2 GPUs as the scaling baseline
- do not introduce CPU offload only for the 1-GPU condition

The compared workloads should remain architecturally consistent.

---

# 18. Strong-Scaling Design

Hold constant:

- model
- precision
- sequence length
- data
- optimizer
- learning-rate schedule
- gradient checkpointing
- optimizer-step count
- global batch size
- per-GPU micro-batch

Change only:

- GPU count
- gradient accumulation needed to keep the global batch fixed

Example:

```text
global batch = 32
micro-batch per GPU = 2
```

| GPUs | Micro-Batch / GPU | Gradient Accumulation | Global Batch |
|---:|---:|---:|---:|
| 1 | 2 | 16 | 32 |
| 2 | 2 | 8 | 32 |
| 4 | 2 | 4 | 32 |

No CPU offload in this scaling series.

---

# 19. Scaling Metrics

Report:

- tokens/sec
- samples/sec
- peak GPU memory
- wall-clock time
- optional communication overhead

If 1 GPU is baseline:

```text
Efficiency(k)
=
throughput(k GPUs)
/
(k × throughput(1 GPU))
```

If 2 GPUs are the baseline:

```text
Relative efficiency(4 vs 2)
=
throughput(4 GPUs)
/
(2 × throughput(2 GPUs))
```

Use a fixed number of optimizer steps after warmup.

Prefer steady-state measurements, such as:

```text
median throughput over final N steps
```

rather than whole-run averages dominated by startup overhead.

---

# 20. Optional LoRA DDP Contrast

Optionally benchmark LoRA under regular DDP.

Question:

> When is simple data-parallel replication sufficient, and when does parameter sharding actually pay off?

This provides a cleaner systems comparison than applying FSDP to workloads that do not need it.

---

# 21. External Data: Conditional Phase 2

Do not add external training data at the beginning.

Trigger Phase 2 only if:

- CV is unstable
- training overfits quickly
- M1/M2 barely improve over B2/B3
- seed SD is comparable to the observed gains

Possible candidate:

```text
HealthVer
```

Before use, verify:

- license
- label definitions
- label mapping
- evidence format
- dataset size
- split integrity
- overlap with SciFact
- domain compatibility

Then compare:

```text
SciFact-only
vs.
SciFact + external data
```

while keeping the same held-out SciFact evaluation.

---

# 22. GRPO: Optional Phase 3

Only attempt GRPO if SFT is already complete and useful.

Example reward:

```text
R =
1[label correct]
+ λ1 * 1[valid output]
+ λ2 * rationale validity
```

with:

```text
λ1 + λ2 < 1
```

so label correctness dominates.

Monitor:

- label collapse
- reward hacking
- output-format shortcuts
- rationale-index gaming

GRPO should not delay the main project.

---

# 23. Timeline

| Week | Work | Exit Criterion |
|---|---|---|
| 0 | Verify held-out split identity and v1 scoring rule. Reproduce B0 | historical baseline reproduced |
| 1 | Run B1-B3 on DEV. Initial audits of Pool B and Pool C (≥50 each). Define each pool's filter, re-audit each (≥50 each). Freeze schema/rationale/calibration scorers | baseline table + both pools' audit rates |
| 2 | Build de-duplicated/filtered training data. Set up grouped CV. First M1 run | full pipeline works |
| 3 | Cross-validate M1/M2. DEV finalists with 3 seeds | resume-ready if result is meaningful |
| 4 | Run M3. FSDP scaling on 2/4 GPUs and 1 GPU if feasible | systems table complete |
| 5 | Calibration, rationale evaluation, efficiency, error analysis. Freeze final configuration | model frozen |
| 6 | Held-out evaluation, stability runs, README, report, resume update | project complete |
| 7+ | Optional external-data or GRPO phase | optional |

---

# 24. Error Analysis

Use categories such as:

```text
retrieval failure
evidence insufficiency
wrong polarity
wrong relation
over-abstention
malformed output
invalid rationale index
rationale mismatch
suspected annotation noise
```

Report counts by model on the primary E1 evaluation.

---

# 25. Success Criteria

## Strong

M2 improves over B0 with:

- positive Macro F1 difference whose paired-bootstrap 95% CI excludes zero
- Accuracy improvement supported by the pre-registered Holm-corrected McNemar analysis
- qualitatively consistent direction across the three training seeds
- lower serving memory and/or latency than B0

---

## Moderate

```text
M2 approximately matches B0
```

but:

- clearly improves over B2 and B3
- costs less to serve than 7B
- shows stable behavior across seeds

This still demonstrates that post-training closes much of the 3B-to-7B gap.

---

## Negative but Informative

Examples:

```text
M1 > B2 only when annotated evidence is supplied
```

or:

```text
M2 improves abstention but lowers SUPPORT recall
```

or:

```text
M3 does not outperform LoRA despite much higher training cost
```

These remain valid project conclusions if reported honestly under the frozen protocol.

---

# 26. Resources and Practical Concerns

Estimated Phase-1 compute:

| Item | Rough Estimate |
|---|---:|
| B0-B3 DEV + held-out | a few GPU-hours |
| LoRA CV + DEV finalists | ~30-60 GPU-hours |
| M3 + scaling runs | ~15-30 GPU-hours |
| Total | ~50-100 GPU-hours |

GPU needs:

- LoRA: 1 GPU with roughly 24-48GB should be sufficient
- M3: preferably 2-4 GPUs with 40GB+ each
- scaling study: ideally one multi-GPU node

Main risks:

- insufficient multi-GPU access
- hard-negative annotation noise
- no measurable SFT gain
- over-abstention
- seed instability
- full fine-tune overfitting

---

# 27. Resume Bullet Template

Use only after results are known.

> **Biomedical Evidence Verification v2:** Post-trained Qwen2.5-3B with LoRA on gold, retrieved, and audited hard-negative scientific evidence, using grouped cross-validation and rationale supervision; compared against 3B and 7B zero-/few-shot baselines with pre-registered paired evaluation on a fixed SciFact benchmark split. Benchmarked full fine-tuning with FSDP across multiple GPUs, reporting accuracy, macro F1, calibration, rationale quality, latency, memory, throughput, and scaling efficiency.

Do not write "official test set" unless verified.

---

# 28. Immediate Next Actions

1. Confirm the origin of the 300-claim held-out split.
2. Recover and document v1's exact scoring rule for non-annotated top-1 evidence.
3. Reproduce B0 in the current environment.
4. Confirm access to a multi-GPU node.
5. Run B1-B3 on DEV.
6. Audit at least 50 candidates from each pool: Pool B (non-annotated top-1) and Pool C (rank 2-10).
7. Define each pool's filter before its re-audit, using a filter type that does not assume candidates rank below gold.
8. Freeze:
   - output schema
   - rationale scorer
   - calibration scorer
   - statistical analysis plan (seed 0 as the tested checkpoint; Holm over tests 1-5)
   - vLLM baseline configuration
9. Start M1.

At this point, the high-level project design should be considered frozen unless Week 0 uncovers a concrete data or scoring inconsistency.
