# Milestone 6 — frozen main benchmark evaluation

> **Disclosure:** This is a fixed 300-claim SciFact benchmark comparison. The split was previously evaluated in v1, and v1 error analysis helped motivate v2. It is not a pristine unseen test set and these results are not an unbiased estimate of generalization. The split was not used for v2 training or model selection; B0 compatibility reproduction was the only permitted pre-freeze access.

No model was trained, no checkpoint was selected from benchmark performance, and no prompt, retrieval, parser, scoring, or decoding setting was changed. M3, FSDP, and serving integration were not run.

## Frozen endpoint and identities

E1—zero-shot BGE exact retrieval over 5,183 SciFact documents followed by the top-1 full abstract—is the sole primary endpoint. All six systems use identical ascending claim order, evidence, prompt/schema, greedy decoding, and claim-level gold scoring. E2–E4 were optional and were not run.

| System | Checkpoint | Few-shot | Frozen config SHA-256 |
|---|---|---:|---|
| B0 | `/home/boweiye2/rag/models/Qwen2.5-7B-Instruct` | no | `6bdea1558da33715f4b417e7216848b68f2905d0e6568d4817bebb1639ae51c2` |
| B1 | `/home/boweiye2/rag/models/Qwen2.5-7B-Instruct` | yes | `64e05069b43c23fd481668c8735c92b6e7f3745b2e2178654af4e78db613d20e` |
| B2 | `/home/boweiye2/rag/models/Qwen2.5-3B-Instruct` | no | `2a4144490a0d21fd4b01675a48ca50f940578c3d7afc87cfbb6ee1f053072cfb` |
| B3 | `/home/boweiye2/rag/models/Qwen2.5-3B-Instruct` | yes | `115e6b961448a6f9288eba6d545eb7b8967b69075b60cce8ead408eb290572ab` |
| M1 | `/home/boweiye2/rag/checkpoints/posttraining/m1/dev-finalists/seed-0/merged` | no | `52f3f84599b80be99fe1fe778986725325db3ee8c71786779187024aebf9c722` |
| M2 | `/home/boweiye2/rag/checkpoints/posttraining/m2/dev-finalists/mix-2-epoch-3/seed-0/merged` | no | `490d28f33dd38b440ca329ed2b56c0ed113d86432dac0ea45085ed12a48c7d09` |

Final evaluation config SHA-256: `6a7c7236ef125132ea3cf8a219d90b2548a288da0a327f14559e20f4d32cc208`. Exact shard hashes are preserved in `preflight.json` and `configs/posttraining/final-eval-v1.json`.

## Main E1 results

| System | Accuracy | Macro F1 | SUPPORT F1/recall | CONTRADICT F1/recall | INSUFFICIENT F1/recall | Predicted S/C/I/invalid |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.6867 | 0.6706 | 0.7064 / 0.6210 | 0.5932 / 0.5469 | 0.7121 / 0.8393 | 94/54/152/0 |
| B1 | 0.7033 | 0.6728 | 0.7554 / 0.7097 | 0.5370 / 0.4531 | 0.7259 / 0.8393 | 109/44/147/0 |
| B2 | 0.6167 | 0.4753 | 0.7071 / 0.8468 | 0.0606 / 0.0312 | 0.6582 / 0.6964 | 173/2/125/0 |
| B3 | 0.6267 | 0.4881 | 0.7323 / 0.7500 | 0.0606 / 0.0312 | 0.6715 / 0.8304 | 130/2/165/3 |
| M1 | 0.4567 | 0.3794 | 0.6842 / 0.6290 | 0.4538 / 0.9219 | 0.0000 / 0.0000 | 104/196/0/0 |
| M2 | 0.6733 | 0.6602 | 0.7149 / 0.6774 | 0.5419 / 0.6562 | 0.7238 / 0.6786 | 111/91/98/0 |

M2 reaches **0.6733 accuracy / 0.6602 macro F1**. Relative to M1 it restores INSUFFICIENT recall from 0.0000 to 0.6786 while retaining much stronger CONTRADICT recall than B2/B3 (0.6562 versus 0.0312). Its 111/91/98 SUPPORT/CONTRADICT/INSUFFICIENT predictions are substantially more balanced than M1 or the untuned 3B baselines.

## Structural validity

| System | JSON | Schema | Decision | Rationale-index | Invalid output |
|---|---:|---:|---:|---:|---:|
| B0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| B1 | 1.0000 | 0.9967 | 1.0000 | 0.9967 | 0.0000 |
| B2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| B3 | 0.9900 | 0.9900 | 0.9900 | 0.9900 | 0.0100 |
| M1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| M2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

B3 produced three invalid outputs. B1 had one structurally valid decision with an invalid rationale index. Structural decision validity and rationale-index validity remain separate.

## Restricted-choice calibration

These are probabilities normalized only across SUPPORT, CONTRADICT, and INSUFFICIENT after scoring `<prompt> + {"decision": " + label`. They are conditional restricted-choice probabilities, not unrestricted generation probabilities. Structured generation remains the official task prediction.

| System | ECE | Brier | NLL | Restricted argmax accuracy | Agreement with generation |
|---|---:|---:|---:|---:|---:|
| B0 | 0.2671 | 0.5598 | 2.4586 | 0.6867 | 0.9733 |
| B1 | 0.2804 | 0.5614 | 2.5228 | 0.6900 | 0.9633 |
| B2 | 0.3287 | 0.7023 | 2.5185 | 0.6067 | 0.9433 |
| B3 | 0.3498 | 0.7161 | 3.7612 | 0.6267 | 0.9733 |
| M1 | 0.2644 | 0.7771 | 2.8946 | 0.5000 | 0.7300 |
| M2 | 0.1468 | 0.4719 | 1.0334 | 0.6900 | 0.8167 |

M1 (0.7300) and M2 (0.8167) fall materially below the predeclared approximately 95% agreement diagnostic. Their calibration path and free structured-generation path therefore differ; calibration values should not be interpreted as confidence in the generated decision.

## Rationale quality

Rationales are scored only where E1 retrieved an annotated document for a true SUPPORT/CONTRADICT claim. For multiple valid sets, the maximum rationale F1 is used.

| System | Eligible / excluded | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| B0 | 149 / 151 | 0.5470 | 0.7002 | 0.5889 |
| B1 | 149 / 151 | 0.5732 | 0.7629 | 0.6230 |
| B2 | 149 / 151 | 0.4092 | 0.7953 | 0.4827 |
| B3 | 149 / 151 | 0.3679 | 0.6499 | 0.4331 |
| M1 | 149 / 151 | 0.8322 | 0.8020 | 0.8110 |
| M2 | 149 / 151 | 0.8121 | 0.7864 | 0.7942 |

Each system has 149 eligible records. The 151 exclusions are 112 gold INSUFFICIENT claims and 39 SUPPORT/CONTRADICT claims whose retrieved top-1 document lacks a valid gold rationale set.

## Pre-registered paired comparisons

Accuracy uses exact two-sided McNemar tests with Holm correction across tests 1–5 only. Macro-F1 intervals are unadjusted paired claim bootstrap percentile intervals with 10,000 resamples. Positive differences favor the left system.

| Test | Comparison | Accuracy Δ | Discordance left-only/right-only | Raw p | Holm p | Macro-F1 Δ | 95% bootstrap CI |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | M1 − B2 | -0.1600 | 58/106 | 2.207e-04 | 8.828e-04 | -0.0959 | [-0.1577, -0.0387] |
| 2 | M2 − B3 | 0.0467 | 47/33 | 0.1456 | 0.4369 | 0.1721 | [0.1131, 0.2291] |
| 3 | M2 − M1 | 0.2167 | 85/20 | 1.008e-10 | 5.042e-10 | 0.2809 | [0.2309, 0.3278] |
| 4 | M2 − B0 | -0.0133 | 34/38 | 0.7239 | 0.7239 | -0.0104 | [-0.0662, 0.0467] |
| 5 | M2 − B1 | -0.0300 | 31/40 | 0.3425 | 0.6849 | -0.0125 | [-0.0724, 0.0467] |

M1 is significantly worse than B2 on accuracy after Holm correction, and its macro-F1 interval is wholly negative. M2 has a clear macro-F1 advantage over B3, although its accuracy difference is not significant after correction. M2 strongly outperforms M1 on both analyses. Against B0 and B1, both M2 confidence intervals include zero and McNemar tests are not significant; this supports statistical comparability on this fixed benchmark rather than superiority or formal equivalence.

## DEV corroboration and seed stability

| System | Frozen DEV accuracy | Frozen DEV macro F1 | Benchmark accuracy | Benchmark macro F1 |
|---|---:|---:|---:|---:|
| B0 | 0.7531 | 0.7476 | 0.6867 | 0.6706 |
| B1 | 0.7531 | 0.7356 | 0.7033 | 0.6728 |
| B2 | 0.6358 | 0.4963 | 0.6167 | 0.4753 |
| B3 | 0.6852 | 0.5388 | 0.6267 | 0.4881 |
| M1 | 0.4753 | 0.3833 | 0.4567 | 0.3794 |
| M2 | 0.6626 | 0.6468 | 0.6733 | 0.6602 |

M1 and M2 DEV values are three-seed means; baseline values are their single frozen runs. For every pre-registered comparison, at least two of three tuned-model DEV seed differences—including seed 0—share the mean difference sign. The frozen seed-consistency rule is satisfied for all five comparisons. DEV and benchmark agree that M1 overpredicts CONTRADICT and never abstains, while M2 restores INSUFFICIENT behavior and retains a large CONTRADICT improvement over B2/B3.

## Technical correction

All six generation passes and B0 calibration completed before the only technical failure. B1 restricted-choice calibration exhausted GPU memory when batch size 6 materialized full-vocabulary logits for long few-shot prompts. No partial B1 calibration file was written. The failed state and traceback context were preserved in `technical-corrections.json`; calibration resumed at B1 with batch size 1. This changes throughput only: prompts, weights, continuations, total sequence likelihoods, normalization, generations, and metrics are unchanged. Generations and B0 calibration were not rerun.

The merged-tokenizer legacy-regex warning seen in prior milestones recurred. Frozen tokenizer behavior was preserved, and zero-shot prompt lengths remained identical across B0, B2, M1, and M2.

## Environment and commands

- Generation: Python 3.11.17, PyTorch 2.7.0+cu126, Transformers 4.53.2, vLLM 0.9.2, CUDA 12.6.
- Calibration/statistics: Python 3.11.16, PyTorch 2.9.1+cu126, Transformers 4.57.6, CUDA 12.6.
- Hardware: one NVIDIA L40S.
- Repository commit recorded before the run: `db145b5e6526afefd7ad9fead083e4100fe53de2`.

```bash
/home/boweiye2/rag/envs/posttraining/bin/python -m pytest -q tests/posttraining tests/test_verification.py tests/test_scifact_preparation.py
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark preflight --config configs/posttraining/final-eval-v1.json
scripts/run_main_benchmark.sh
# After the documented B1 calibration OOM, resume only unfinished calibration with --batch-size 1
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system B1
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system B2 --system B3
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system M1
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark calibrate --batch-size 1 --config configs/posttraining/final-eval-v1.json --system M2
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.benchmark finalize --config configs/posttraining/final-eval-v1.json
```

## Artifacts

- External run: `/home/boweiye2/rag/runs/posttraining/benchmark-eval-v1`
- Per-system predictions, raw generations, calibration scores, metrics, and validity: `/home/boweiye2/rag/runs/posttraining/benchmark-eval-v1/systems/{B0,B1,B2,B3,M1,M2}/`
- Paired statistics and bootstrap distributions: `/home/boweiye2/rag/runs/posttraining/benchmark-eval-v1/statistics/`
- Technical correction: `/home/boweiye2/rag/runs/posttraining/benchmark-eval-v1/technical-corrections.json`
- Pre-benchmark snapshot: `/home/boweiye2/rag/runs/posttraining/benchmark-eval-v1/source/pre-benchmark-source-config-dependencies.tar.gz`
- Pre-benchmark snapshot SHA-256: `6818900d242f0ecd2da5d72f0d54fe3c9510539849843e6a166dd6d8ae6e17c3`
- Repository report results: `docs/posttraining/milestone6-main-benchmark-results.json`

## Interpretation and stop point

The fixed benchmark answers the five questions as follows: M1 does not beat B2; M2 improves substantially over B3 in macro F1; M2 clearly beats M1; and M2 performs near B0/B1 without a statistically reliable difference in either direction under the pre-registered analyses. This evidence is descriptive of a reused benchmark and does not establish unbiased generalization or formal model equivalence.

The main benchmark passed. No implementation blocker remains before a separately authorized M3/FSDP stage. Milestone 6 stops here; no M3, retraining, post-benchmark tuning, or serving integration was performed.
