# Milestone 4 — M2 mixed-evidence LoRA SFT

> **Scope:** M2 trains Qwen2.5-3B-Instruct with frozen LoRA settings on deterministic TRAIN-only evidence mixtures. Mixture and epoch candidates are selected by grouped five-fold TRAIN CV. Two CV finalists are each trained with three predefined seeds and evaluated on the frozen 162-claim DEV protocol. No fixed 300-claim benchmark record was accessed; M3, FSDP, and serving integration were not run.

## Outcome

M2 passed implementation and execution validation. The selected system is **Mix-2, three epochs**. Across seeds 0–2 it reaches **0.6626 ± 0.0128 accuracy** and **0.6468 ± 0.0107 macro F1** on DEV.

The main diagnostic succeeds under this setup:

- INSUFFICIENT recall is restored from M1's 0.0000 to **0.5625 ± 0.0156**.
- CONTRADICT recall remains high at **0.7738 ± 0.0206**, versus 0.0714 for B2 and B3.

Mixed-evidence training therefore corrects M1's collapse of abstention while retaining most of its CONTRADICT gain. M2 substantially outperforms M1 and exceeds B3 macro F1, although its DEV accuracy remains below B3 and both 7B baselines.

## Frozen source pools and construction

| Pool | Meaning | Available | SHA-256 |
|---|---|---:|---|
| A | annotated gold evidence | 456 | `2f4397bcd42f845dba64445d5d63864f2b0e620f3982134bfe11010d3a39f590` |
| B | approved retrieved non-gold top-1 | 86 | `29b52fd103d54459f23bdae664ed5436d26f5d943e868408ee357a23b0142f3e` |
| C | approved rank 2–10 hard negatives | 5,716 | `01f26d469d9975c643ff7dbe84b0fde594cc9ff89db7cd97bd195bb5de4aa628` |
| D | deterministic random negatives | 647 | `dda762c06c238ec032c043296c5af7d0749fe7188c93bde34367de0944a28c5b` |

Pool C passed its frozen audit and was retained, so the Mix-3 fallback was not run. Pool B limits the largest exact-proportion datasets. The frozen construction takes the largest total permitted without oversampling, applies Hamilton rounding, and samples deterministically within each pool.

| Candidate | Intended A/B/C/D | Effective counts A/B/C/D | Effective percentages | Rows | Sampling seed | Dataset SHA-256 |
|---|---|---|---|---:|---:|---|
| Mix-1 | 45/20/20/15% | 194 / 86 / 86 / 64 | 45.12 / 20.00 / 20.00 / 14.88% | 430 | 20261020 | `66c90244c238722467a362435c79bf0a6bffe65a8d255a0230d0a891530df7e9` |
| Mix-2 | 60/15/15/10% | 344 / 86 / 86 / 57 | 60.03 / 15.01 / 15.01 / 9.95% | 573 | 20261021 | `7ca1013d3ff393eddb805f0e3ce5b5dc85b18055775c264dcdc073867514dfeb` |

Mix-1 contains 120 SUPPORT, 74 CONTRADICT, and 236 INSUFFICIENT targets. Mix-2 contains 219 SUPPORT, 125 CONTRADICT, and 229 INSUFFICIENT targets. Both have zero duplicate claim-document pairs. All A targets equal their exact pair annotations and retain annotated rationale indices. All B/C/D targets are INSUFFICIENT with empty rationales. Every row is from TRAIN and retains source provenance, mixture name, and sampling seed.

An initial manifest-only counter incorrectly printed one row per pool because it counted a set of pool names. This was caught before training. The counter was corrected, both reconstructed JSONL files were proven byte-identical to the originals, and only the invalid manifest was replaced.

## Frozen training configuration

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Model/tokenizer revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| LoRA targets | `q_proj`, `v_proj` |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Trainable / total parameters | 3,686,400 / 3,089,625,088 |
| Trainable percentage | 0.1193% |
| Precision | BF16 |
| Microbatch / accumulation / effective batch | 2 / 8 / 16 |
| Optimizer | AdamW, LR 2e-4, weight decay 0.01 |
| Scheduler | linear over three-epoch horizon, 10% warmup |
| Gradient clipping | 1.0 |
| Maximum sequence length | 4,096 |
| Candidate epochs | 2, 3 |
| CV seed | 20261019 |
| Final seeds | 0, 1, 2 |

The longest rendered sequence is 1,923 tokens. Prompt tokens remain masked with `-100`; only structured answer tokens contribute to SFT loss. A mixed SUPPORT/INSUFFICIENT smoke batch produced finite gradients and updated a LoRA parameter.

## Grouped TRAIN cross-validation

One shared fold assignment covers all 647 frozen TRAIN claim IDs exactly once. Every row from a claim stays in that claim's fold across both mixtures. Mix-1 validation row counts are 84, 88, 81, 90, and 87; Mix-2 counts are 120, 110, 121, 111, and 111.

| Candidate | Mean accuracy | Mean macro F1 | SUPPORT recall | CONTRADICT recall | INSUFFICIENT recall | Cumulative seconds | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mix-1, epoch 2 | 0.8149 | 0.7547 | 0.6584 | 0.6994 | 0.9270 | 550.6 | 0.1529 |
| Mix-1, epoch 3 | 0.8195 | 0.7564 | 0.6679 | 0.6496 | 0.9489 | 824.0 | 0.2289 |
| Mix-2, epoch 2 | 0.7899 | 0.7707 | 0.6615 | 0.7749 | 0.9187 | 751.5 | 0.2087 |
| Mix-2, epoch 3 | **0.8004** | **0.7804** | 0.6501 | 0.7848 | 0.9532 | 1,126.7 | 0.3130 |

Mix-2 epoch 3 and Mix-2 epoch 2 were the top two CV configurations and were frozen as DEV finalists. Mix-2 epoch 3 exceeds epoch 2 by 0.0097 mean macro F1, outside the 0.005 tie threshold. Mix-1 epoch 2 is ranked ahead of Mix-1 epoch 3 under the tie rule because their 0.0018 difference is within the threshold.

The epoch-2 timing values are cumulative through epoch 2 in the same CV trajectories. Actual CV training compute is the sum of the two complete three-epoch candidate-mixture trajectories: 1,950.7 seconds.

## DEV finalist results

All evaluations use the matched B2 protocol: frozen zero-shot BGE top-1 full abstract, the corrected zero-shot prompt, merged adapters, vLLM 0.9.2, greedy decoding, repetition penalty 1.05, and a 192-token cap.

| Finalist | Seed | Accuracy | Macro F1 | SUPPORT recall | CONTRADICT recall | INSUFFICIENT recall | Predictions S/C/I |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mix-2 epoch 2 | 0 | 0.6111 | 0.6014 | 0.5857 | 0.7857 | 0.5625 | 54 / 60 / 48 |
| Mix-2 epoch 2 | 1 | 0.6543 | 0.6427 | 0.6429 | 0.7857 | 0.6094 | 61 / 51 / 50 |
| Mix-2 epoch 2 | 2 | 0.5988 | 0.5908 | 0.5714 | 0.7857 | 0.5469 | 54 / 65 / 43 |
| **Epoch-2 mean ± SD** | — | **0.6214 ± 0.0292** | **0.6116 ± 0.0274** | **0.6000 ± 0.0378** | **0.7857 ± 0.0000** | **0.5729 ± 0.0325** | — |
| Mix-2 epoch 3 | 0 | 0.6728 | 0.6568 | 0.7286 | 0.7857 | 0.5625 | 67 / 49 / 46 |
| Mix-2 epoch 3 | 1 | 0.6481 | 0.6356 | 0.6571 | 0.7857 | 0.5781 | 62 / 53 / 47 |
| Mix-2 epoch 3 | 2 | 0.6667 | 0.6479 | 0.7429 | 0.7500 | 0.5469 | 70 / 48 / 44 |
| **Epoch-3 mean ± SD** | — | **0.6626 ± 0.0128** | **0.6468 ± 0.0107** | **0.7095 ± 0.0459** | **0.7738 ± 0.0206** | **0.5625 ± 0.0156** | — |

Epoch 3 is selected using the frozen best-three-seed-mean macro-F1 rule. Seed 0 is the predeclared future inferential checkpoint; its own DEV score did not choose the finalist. Across all 972 finalist outputs, JSON, decisions, and rationale indices are 100% valid.

Selected epoch-3 mean class F1 is 0.7281 ± 0.0270 SUPPORT, 0.5558 ± 0.0144 CONTRADICT, and 0.6565 ± 0.0094 INSUFFICIENT.

## DEV comparisons

| Comparison | Accuracy difference | Macro-F1 difference |
|---|---:|---:|
| M2 − M1 | **+0.1872** | **+0.2635** |
| M2 − B3 | −0.0226 | **+0.1079** |
| M2 − B0 | −0.0905 | −0.1008 |
| M2 − B1 | −0.0905 | −0.0889 |

These are descriptive DEV comparisons. No fixed-benchmark prediction or inferential test was run.

## Compute

| Finalist | Seed | Seconds | GPU-hours | Steps | Tokens | Tokens/s | Samples/s | Mean step s | Peak PyTorch MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch 2 | 0 | 189.9 | 0.0527 | 72 | 746,060 | 3,929.1 | 6.0354 | 2.6366 | 13,385.7 |
| Epoch 2 | 1 | 190.7 | 0.0530 | 72 | 746,060 | 3,911.6 | 6.0084 | 2.6485 | 13,385.7 |
| Epoch 2 | 2 | 190.1 | 0.0528 | 72 | 746,060 | 3,924.6 | 6.0285 | 2.6396 | 13,393.4 |
| Epoch 3 | 0 | 284.4 | 0.0790 | 108 | 1,119,090 | 3,934.5 | 6.0437 | 2.6330 | 13,385.7 |
| Epoch 3 | 1 | 285.2 | 0.0792 | 108 | 1,119,090 | 3,924.0 | 6.0275 | 2.6401 | 13,385.7 |
| Epoch 3 | 2 | 284.2 | 0.0790 | 108 | 1,119,090 | 3,937.1 | 6.0477 | 2.6313 | 13,393.4 |

- CV training: 1,950.7 seconds / 0.5418 GPU-hours.
- Six finalist runs: 1,424.6 seconds / 0.3957 GPU-hours.
- Total M2 training: **3,375.2 seconds (56.25 minutes) / 0.9376 GPU-hours**.
- Peak PyTorch allocation: **13,393.4 MiB**.

Evaluation time, model merging, and vLLM startup are excluded from training compute.

## Validation and deviations

- Focused tests were run before construction and before CV; the final combined posttraining/data/verification suite passed **56 tests** after finalist freezing.
- All mixture hashes, source hashes, claim-group folds, selected configurations, predictions, and merged model shards reconstruct successfully.
- Completed run and checkpoint directories cannot be overwritten.
- The client-side Transformers tokenizer emitted its known legacy-regex warning. The protocol was not changed; prompt token counts exactly match B2 (mean 579.586, maximum 1,189), and vLLM used the saved local tokenizer for generation.
- vLLM's absolute local-path metadata lookup warning fell back to direct local safetensors loading for every model.
- The requested GPT model choice is controlled by the Codex runtime rather than repository code and has no effect on training or evaluation.

## Commands

```bash
source scripts/env.sh
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.data.build_m2_mixtures

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.training.m2_mixed preflight

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.training.m2_mixed cv

for epochs in 3 2; do
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      /home/boweiye2/rag/envs/posttraining/bin/python \
      -m posttraining.training.m2_mixed final \
      --mixture mix-2 --epochs "$epochs" --seed "$seed"
  done
done

scripts/run_m2_dev_evaluations.sh
/home/boweiye2/rag/envs/posttraining/bin/python -m posttraining.evaluation.finalize_m2
```

## Changed repository files

- `configs/posttraining/m2-mixtures-v1.json`
- `configs/posttraining/train-m2-lora-mixed-v1.json`
- `configs/posttraining/m2-cv-folds-v1.json`
- `configs/posttraining/m2-cv-finalists-v1.json`
- `configs/posttraining/m2-finalist-v1.json`
- six `configs/posttraining/m2-mix-2-epoch-*-seed-*-dev-v1.json` files
- `posttraining/data/build_m2_mixtures.py`
- `posttraining/training/m2_mixed.py`
- `posttraining/evaluation/finalize_m2.py`
- `scripts/run_m2_dev_evaluations.sh`
- `tests/posttraining/test_m2_mixtures.py`
- `tests/posttraining/test_m2_training.py`
- `environments/posttraining-m2-pip-freeze.txt`
- this report and its machine-readable results JSON

## Artifacts

- Mixed datasets: `/home/boweiye2/rag/data/posttraining/mixtures-v1/`
- Runs, predictions, and metrics: `/home/boweiye2/rag/runs/posttraining/lora-mixed-v1/`
- Adapters and merged models: `/home/boweiye2/rag/checkpoints/posttraining/m2/`
- vLLM logs: `/home/boweiye2/rag/logs/m2-mix-2-epoch-*-seed-*-vllm.log`
- Machine-readable repository results: `docs/posttraining/milestone4-m2-lora-mixed-results.json`
- Exact snapshot: `/home/boweiye2/rag/runs/posttraining/lora-mixed-v1/source/exact-source-config-dependencies.tar.gz`

M2 stops here. The selected inferential checkpoint is Mix-2 epoch 3 seed 0. There is no implementation blocker before a separately authorized frozen benchmark stage.
