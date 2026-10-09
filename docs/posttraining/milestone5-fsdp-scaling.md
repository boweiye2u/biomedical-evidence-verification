# Milestone 5 — FSDP strong scaling

M3 full-parameter fine-tuning used PyTorch FSDP `FULL_SHARD`, Qwen2 decoder-layer auto-wrapping, BF16 parameter/reduction/buffer compute, FP32 master parameters and AdamW state, activation checkpointing, and no CPU offload. The frozen global batch was 16 with microbatch 2 per GPU.

## Fit and scaling results

| GPUs | Fit | Accumulation | Measured seconds (steps 3–12) | Tokens/s | Samples/s | GPU-hours (whole run) | Peak MiB/GPU |
|---:|:---:|---:|---:|---:|---:|---:|---|
| 1 | No | 8 | — | — | — | — | ~45,434 before OOM |
| 2 | Yes | 4 | 33.79 | 3106.85 | 4.734 | 0.02520 | 29590, 29568 |
| 4 | Yes | 2 | 49.52 | 2120.23 | 3.231 | 0.07471 | 20256, 20153, 19226, 19708 |

The one-GPU FSDP attempt first exposed a world-size-one mixed-precision gradient-view incompatibility. A controlled native single-GPU retry retained FP32 master parameters and BF16 autocast, then produced a genuine CUDA OOM at the first AdamW state allocation: only about 38 MiB remained on a 46,068 MiB L40S when another 86 MiB was requested. No CPU offload or altered scientific settings were used.

With two GPUs as the baseline, four-GPU speedup was **0.682×** and strong-scaling efficiency was **0.341**. Four GPUs were slower for this small model and batch because added communication outweighed sharding gains. Two GPUs gave the best measured throughput; four GPUs reduced peak memory and was retained for the predeclared scientific jobs.

## Reproducibility notes

Both successful runs executed 12 optimizer steps over the same 192 samples and 126,228 tokens with identical model, revision, Mix-2 hash, optimizer, LR schedule, sequence limit, seed, and global batch. NCCL 2.27.5 was used through PyTorch 2.9.1+cu126. The raw scaling files were produced before a reporting correction and record local-shard `trainable_parameter_count`; the correct global count is **3,085,938,688**, also preserved in `parameter_count`.

Artifacts are under `/home/boweiye2/rag/runs/posttraining/fsdp-scaling-v1/`; logs are under `/home/boweiye2/rag/logs/m3-fsdp-scaling-*`.
