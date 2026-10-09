# Milestone 3 — M1 gold-evidence LoRA SFT

> **Scope:** M1 trains Qwen2.5-3B-Instruct with LoRA using only frozen Condition A gold-evidence TRAIN pairs. Epoch selection uses grouped five-fold TRAIN cross-validation. The selected configuration is then trained with three predefined seeds and evaluated once per seed on the frozen 162-claim DEV protocol. No fixed 300-claim benchmark record was accessed, and M2 was not constructed or trained.

## Outcome

M1 **passed implementation and execution validation**, but its scientific result is negative on the three-class DEV task. Gold-only SFT substantially increased CONTRADICT recall while eliminating INSUFFICIENT predictions. Across three seeds, M1 reaches **0.4753 ± 0.0283 accuracy** and **0.3833 ± 0.0171 macro F1**, below B2 by 0.1605 accuracy and 0.1130 macro F1.

This is the intended clean control. Condition A contains SUPPORT and CONTRADICT examples only, so the result shows the cost of gold-only adaptation under this training setup. It does not show that LoRA or supervised fine-tuning is generally harmful.

## Frozen data and configuration

The input is `/home/boweiye2/rag/data/posttraining/train_gold.jsonl`, SHA-256 `2f4397bcd42f845dba64445d5d63864f2b0e620f3982134bfe11010d3a39f590`:

- 456 Condition A TRAIN claim-document pairs
- 292 SUPPORT and 164 CONTRADICT
- 407 unique `claim_id` groups
- no Pool B, C, or D rows
- no DEV rows and no fixed-benchmark rows
- `pair_annotation == context_label` for every row
- valid, document-bound rationale indices for every row

The longest rendered training sequence is 1,753 tokens and the longest answer is 21 tokens, below the fixed 4,096-token limit. Prompt tokens are masked with `-100`; only assistant answer tokens contribute to causal-LM loss. A real two-example smoke update produced finite loss and gradients and changed a LoRA parameter before full training.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Model/tokenizer revision | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| LoRA targets | `q_proj`, `v_proj` |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Trainable parameters | 3,686,400 |
| Total parameters during PEFT training | 3,089,625,088 |
| Trainable percentage | 0.1193% |
| Precision | BF16 |
| Microbatch / accumulation / effective batch | 2 / 8 / 16 |
| Optimizer | AdamW, LR 2e-4, weight decay 0.01 |
| Scheduler | linear, 10% warmup |
| Gradient clipping | 1.0 |
| Gradient checkpointing | non-reentrant |
| Candidate epochs | 2, 3 |
| CV seed | 20261019 |
| Final seeds | 0, 1, 2 |
| GPU | one NVIDIA L40S |

The attention projection targets were frozen before training as the smallest standard LoRA set present throughout the Qwen architecture. Software versions were PyTorch 2.9.1+cu126, CUDA runtime 12.6, Transformers 4.57.6, PEFT 0.21.2, and TRL 1.15.0. The custom loop uses Transformers and PEFT directly; TRL is recorded but is not used by the runner.

## Grouped TRAIN cross-validation

All rows from one claim remain in one fold. Fold validation row counts are 92, 91, 91, 91, and 91; validation claim counts are 82, 81, 81, 81, and 82. The five validation claim sets are disjoint and cover all 407 groups.

| Epoch | Fold macro F1 | Mean macro F1 | Fold accuracy | Mean accuracy | Cumulative training time | Cumulative GPU-hours |
|---:|---|---:|---|---:|---:|---:|
| 2 | 0.4637, 0.4573, 0.5340, 0.5109, 0.4903 | 0.4912 | 0.6957, 0.6923, 0.8022, 0.7692, 0.7363 | 0.7391 | 623.6 s | 0.1732 |
| 3 | 0.4850, 0.4680, 0.5728, 0.5250, 0.5541 | **0.5210** | 0.7283, 0.7033, 0.8681, 0.7912, 0.8352 | **0.7852** | 935.8 s | 0.2599 |

Epoch 3 is selected because its mean macro F1 is 0.0297 higher than epoch 2, exceeding the frozen 0.005 tie margin. DEV did not influence this choice. The epoch-2 time is the cumulative cost through epoch 2 within the same trajectories; actual CV compute is the epoch-3 total.

| Fold | Rows trained | Steps | Training seconds | GPU-hours | Peak PyTorch MiB | Tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 364 | 69 | 185.0 | 0.0514 | 12,729.2 | 3,999.9 |
| 1 | 365 | 69 | 188.0 | 0.0522 | 12,740.0 | 3,945.7 |
| 2 | 365 | 69 | 189.4 | 0.0526 | 12,745.5 | 3,969.1 |
| 3 | 365 | 69 | 188.2 | 0.0523 | 12,740.1 | 3,928.9 |
| 4 | 365 | 69 | 185.3 | 0.0515 | 12,740.0 | 4,026.5 |

CV evaluation used deterministic Transformers greedy generation on held-out TRAIN folds. Final DEV confirmation used merged models through the same vLLM 0.9.2 protocol as B2.

## Three-seed DEV confirmation

Each seed independently reloaded the pinned base checkpoint and recreated its optimizer, scheduler, and random state. Each trained for exactly three epochs and 87 optimizer steps. DEV uses frozen zero-shot BGE top-1 full abstracts, the corrected zero-shot prompt shared with B2, greedy vLLM decoding, repetition penalty 1.05, and a 192-token output cap.

| Seed | Accuracy | Macro F1 | SUPPORT F1 / recall | CONTRADICT F1 / recall | INSUFFICIENT F1 / recall | Predictions S / C / I |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4691 | 0.3803 | 0.7376 / 0.7429 | 0.4034 / 0.8571 | 0.0000 / 0.0000 | 71 / 91 / 0 |
| 1 | 0.5062 | 0.4016 | 0.7867 / 0.8429 | 0.4182 / 0.8214 | 0.0000 / 0.0000 | 80 / 82 / 0 |
| 2 | 0.4506 | 0.3679 | 0.7101 / 0.7000 | 0.3934 / 0.8571 | 0.0000 / 0.0000 | 68 / 94 / 0 |
| **Mean ± sample SD** | **0.4753 ± 0.0283** | **0.3833 ± 0.0171** | **0.7448 ± 0.0388 / 0.7619 ± 0.0733** | **0.4050 ± 0.0125 / 0.8452 ± 0.0206** | **0.0000 ± 0.0000 / 0.0000 ± 0.0000** | — |

All 486 outputs across the three seeds have valid JSON, a valid decision, and valid rationale indices. There were no invalid outputs. The stable absence of INSUFFICIENT predictions across all seeds is the main failure mode.

## M1 versus B2

| System | Accuracy | Macro F1 | CONTRADICT recall |
|---|---:|---:|---:|
| B2 — 3B zero-shot | 0.6358 | 0.4963 | 0.0714 |
| M1 — three-seed mean | 0.4753 | 0.3833 | 0.8452 |
| M1 − B2 | **−0.1605** | **−0.1130** | **+0.7738** |

M1 corrects the 3B baseline's severe CONTRADICT underprediction, but the binary-label training support shifts the model away from abstention. The resulting loss on 64 INSUFFICIENT DEV claims outweighs the CONTRADICT gain. This comparison is DEV/CV evidence only; no fixed-benchmark inferential test was run.

## Training efficiency

| Seed | Training seconds | GPU-hours | Steps | Tokens | Tokens/s | Samples/s | Mean step s | Peak PyTorch MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 236.9 | 0.0658 | 87 | 929,679 | 3,924.5 | 5.7748 | 2.7222 | 12,744.9 |
| 1 | 238.4 | 0.0662 | 87 | 929,679 | 3,899.9 | 5.7386 | 2.7394 | 12,741.7 |
| 2 | 238.5 | 0.0662 | 87 | 929,679 | 3,898.3 | 5.7363 | 2.7405 | 12,741.0 |

Actual training compute was 935.8 seconds for CV plus 713.8 seconds for the three final runs: **1,649.6 seconds (27.49 minutes) and 0.4582 single-GPU hours**. Evaluation and model-server startup are excluded from training time.

## Validation and implementation notes

- The focused suite passed before CV, before final training, and after freezing the finalist; final validation passed all 44 requested related tests.
- The runner refuses existing run and checkpoint directories.
- Every retained adapter is associated with config hash, data hash, seed, epoch/step, base revisions, and the Git commit available at execution. Fold identity is encoded in its versioned path.
- Each final adapter was merged into a standalone two-shard model before vLLM evaluation.
- The client-side Transformers tokenizer emitted its known legacy-regex warning for the saved Qwen tokenizer. The frozen protocol was not changed; prompt token counts exactly match B2 (mean 579.586, maximum 1,189), and vLLM performed generation with the saved local tokenizer.
- vLLM logged a failed Hugging Face metadata lookup for each absolute local model path, then loaded both local safetensors shards successfully. This did not change inference.
- CV was run immediately before timing aliases and checkpoint-side manifests were added to the runner. Scientific computation was unchanged; the missing reporting fields were reconstructed from the original immutable CV metrics and paths. Final-seed runs used the reported runner.

## Commands

```bash
source scripts/env.sh

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.training.m1_gold preflight

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.training.m1_gold cv

for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/boweiye2/rag/envs/posttraining/bin/python \
    -m posttraining.training.m1_gold final --seed "$seed"
done

# Repeated for seed 0, 1, and 2 with the corresponding merged path/name.
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/boweiye2/rag/envs/serving/bin/vllm serve "$MERGED" \
  --served-model-name "$MODEL_NAME" --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.72 --max-num-seqs 16 --generation-config vllm \
  --host 127.0.0.1 --port 8001 --disable-log-requests

/home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.evaluation.baselines \
  --config "$CONFIG" \
  --mapping /home/boweiye2/rag/runs/milestone5a-verification-v1/verification-mapping.json \
  --rankings /home/boweiye2/rag/runs/scifact_dev_v1/bge-rankings.json \
  --corpus /home/boweiye2/rag/data/scifact/original/corpus.jsonl \
  --tokenizer "$MERGED" --output-dir "$OUTPUT" \
  --base-url http://127.0.0.1:8001

/home/boweiye2/rag/envs/posttraining/bin/python \
  -m posttraining.evaluation.finalize_m1
```

## Artifacts

Repository:

- training config: `configs/posttraining/train-m1-lora-gold-v1.json`
- grouped folds: `configs/posttraining/m1-gold-cv-folds-v1.json`
- pre-DEV finalist freeze: `configs/posttraining/m1-gold-finalist-v1.json`
- seed evaluation configs: `configs/posttraining/m1-gold-seed-{0,1,2}-dev-v1.json`
- runner: `posttraining/training/m1_gold.py`
- validator/aggregator: `posttraining/evaluation/finalize_m1.py`
- machine-readable results: `docs/posttraining/milestone3-m1-lora-gold-results.json`
- dependency freeze: `environments/posttraining-m1-pip-freeze.txt`

External:

- complete metrics and predictions: `/home/boweiye2/rag/runs/posttraining/lora-gold-v1/`
- adapters and merged models: `/home/boweiye2/rag/checkpoints/posttraining/m1/`
- exact final source/config/dependency snapshot: `/home/boweiye2/rag/runs/posttraining/lora-gold-v1/source/exact-source-config-dependencies.tar.gz`

There is no implementation blocker before a separately authorized M2. M1 stops here. M2 data was not assembled and no M2, M3, FSDP, or fixed-benchmark work was run.
