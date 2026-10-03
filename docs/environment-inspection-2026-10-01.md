# Environment inspection — 2026-10-01 UTC

## Observed

- Linux x86_64, kernel 5.15.0-133-generic.
- Active interpreter: Anaconda base Python 3.12.7.
- Eight NVIDIA L40S GPUs, each reporting 46068 MiB memory. No GPU processes
  were reported at inspection time; this does not establish exclusive allocation.
- NVIDIA driver: 565.57.01. `nvidia-smi` advertises CUDA support through 12.7.
- Installed CUDA toolkit (`nvcc`): 12.2.140. This is distinct from the driver
  capability and from any future PyTorch wheel's bundled CUDA runtime.
- RAM: approximately 1.0 TiB total, 996 GiB available at inspection.
- Root filesystem: approximately 1.8 TB total, 1.5 TB available; both storage
  locations are on this filesystem.
- Active environment contains NumPy 1.26.4, SciPy 1.13.1, pytest 7.4.4.
- PyTorch, Transformers, Sentence Transformers, BEIR, datasets, huggingface-hub,
  vLLM, rank-bm25, faiss-cpu, and FastAPI were not installed in active Python.
- Other environment directories: EVO2, deeptools, epcot. Their packages were
  not inspected and these environments were not modified.
- Existing Hugging Face cache directory: `~/.cache/huggingface/hub`; cached model
  contents were not inventoried or moved.
- `~/rag` existed and was empty before project storage directories were created.
- Workspace is not a Git repository. No applicable AGENTS.md was found during
  the workspace/ancestor inspection.

## Changes

Added a session-local cache/path setup script, Git exclusions, documentation,
and external artifact directories. No packages, models, or datasets were downloaded.
No global shell configuration was changed. Resume files were left in place.

## Blockers and unresolved checks

- Missing project dependencies prevent model inference and dataset validation.
- PyTorch CUDA access remains unverified because PyTorch is absent in base.
- Download connectivity, annotation mappings, and model compatibility are untested.
- The command sandbox fails at startup with `mountinfo path is not absolute`;
  inspection and setup commands required approved execution outside the sandbox.

## Next bounded milestone

Create an isolated retrieval environment under `~/rag/envs` after selecting compatible
package versions. Keep any later vLLM environment separate. Source `scripts/env.sh`
before installations/downloads. Use one explicitly selected GPU for the study.

Then verify BEIR/original SciFact mappings, create a deterministic grouped train/dev
split, validate BGE and MedCPT adapters, verify metrics, and run BM25 plus zero-shot
BGE on DEV against the full document corpus. Do not evaluate test queries or start
fine-tuning, generation, or service implementation in this milestone.
