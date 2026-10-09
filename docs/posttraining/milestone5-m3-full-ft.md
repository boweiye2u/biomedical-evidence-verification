# Milestone 5 — M3 full-parameter fine-tuning

M3 asks whether full-parameter Qwen2.5-3B-Instruct fine-tuning improves enough over the frozen M2 LoRA system to justify its compute and systems cost. It uses the identical frozen Mix-2 data and no benchmark examples for training or selection.

## Frozen configuration

- Mix-2: 573 rows (A 344 / B 86 / C 86 / D 57), SHA-256 `7ca1013d3ff393eddb805f0e3ce5b5dc85b18055775c264dcdc073867514dfeb`
- Model/tokenizer revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Trainable parameters: **3,085,938,688**; no LoRA
- AdamW, LR 1e-5, weight decay 0.01, linear schedule, 10% warmup, BF16 compute with FP32 masters, maximum length 4,096, output-only loss
- FSDP FULL_SHARD on four L40S GPUs, microbatch 2/GPU, accumulation 2, global batch 16, no CPU offload

## Grouped cross-validation

| Epoch | Fold macro F1 | Mean macro F1 | Mean accuracy |
|---:|---|---:|---:|
| 2 | 0.8040, 0.8106, 0.7869, 0.8397, 0.7869 | 0.8056 | 0.8238 |
| 3 | 0.8179, 0.9199, 0.8237, 0.8835, 0.8031 | 0.8496 | 0.8614 |

Both predeclared finalists advanced to the frozen three-seed DEV comparison.

## Frozen DEV results

| Epoch | Seed | Accuracy | Macro F1 | SUPPORT recall | CONTRADICT recall | INSUFFICIENT recall | JSON valid | Decision valid | Rationale-index valid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 0.7160 | 0.6885 | 0.8286 | 0.6429 | 0.6250 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1 | 0.6975 | 0.6715 | 0.8857 | 0.8214 | 0.4375 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 2 | 0.6852 | 0.6670 | 0.8286 | 0.8929 | 0.4375 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 0 | 0.7346 | 0.7158 | 0.8286 | 0.7500 | 0.6250 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1 | 0.6728 | 0.6526 | 0.8286 | 0.7857 | 0.4531 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 2 | 0.7222 | 0.7035 | 0.8143 | 0.7500 | 0.6094 | 1.0000 | 1.0000 | 1.0000 |

| Epoch | Mean ± SD accuracy | Mean ± SD macro F1 |
|---:|---:|---:|
| 2 | 0.6996 ± 0.0155 | 0.6757 ± 0.0114 |
| 3 | 0.7099 ± 0.0327 | 0.6906 ± 0.0335 |

The frozen rule selected **epoch 3**. Seed 0 is the predeclared inferential checkpoint: `/home/boweiye2/rag/checkpoints/posttraining/m3/dev-finalists/seed-0/epoch-3/model`.

Selected three-seed DEV comparison:

| Metric | M2 LoRA mean | M3 full-FT mean | M3 − M2 |
|---|---:|---:|---:|
| Accuracy | 0.6626 | 0.7099 | +0.0473 |
| Macro F1 | 0.6468 | 0.6906 | +0.0439 |
| SUPPORT recall | 0.7095 | 0.8238 | +0.1143 |
| CONTRADICT recall | 0.7738 | 0.7619 | −0.0119 |
| INSUFFICIENT recall | 0.5625 | 0.5625 | 0.0000 |

## Fixed benchmark and M3 versus M2

M3 reached **0.7133 accuracy** and **0.7018 macro F1** on the fixed 300-claim SciFact benchmark split, previously evaluated in v1 and not used for M3 training or model selection.

Relative to the saved M2 predictions, accuracy changed by **+0.0400** and macro F1 by **+0.0416** (paired bootstrap 95% CI [-0.0119, 0.0958]). Exact paired McNemar p = **0.1686**; test 6 is separate from the original Holm family.

Calibration: ECE 0.2071, Brier 0.4919, NLL 1.2198, restricted-choice/generation agreement 0.9800. Rationale precision/recall/F1: 0.8121/0.7897/0.7964 over 149 eligible examples.

## Compute and interpretation

Cross-validation plus final training took 1.07 aggregate wall-clock hours and **4.26 GPU-hours**. Peak PyTorch allocation was **22893 MiB/GPU**; final-run mean throughput was **2068.4 tokens/s**.

On DEV, selected M3 minus M2 was +0.0473 accuracy and +0.0439 macro F1. The benchmark comparison determines whether this translated into a reliable practical gain. M3 used **4.26 GPU-hours**, about **4.5×** M2’s 0.94 GPU-hours, and required multi-GPU sharding because one GPU did not fit. The benchmark point estimates favor M3, but the macro-F1 interval crosses zero and McNemar p=0.1686. **The present evidence does not justify making full fine-tuning the default over M2 LoRA**; M3 is a promising secondary result, while M2 remains the more efficient supported choice.

A four-rank distributed sampler pads 573 rows to 576 samples per epoch, repeating three existing rows; it introduces no new source examples. The fixed benchmark is not described as pristine unseen data. No serving integration was performed.
