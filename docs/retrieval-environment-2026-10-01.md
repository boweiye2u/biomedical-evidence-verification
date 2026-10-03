# Retrieval environment verification — 2026-10-01

Installed at `~/rag/envs/retrieval` using Conda-forge Python and pip packages.
Conda base and existing environments were not modified. Conda/pip caches and
installation logs are under `~/rag`. No datasets or model weights were downloaded.

## Verified versions

- Python 3.11.16
- PyTorch 2.9.1+cu126 (bundled CUDA runtime 12.6)
- Transformers 4.57.6
- Sentence Transformers 5.7.0
- datasets 4.8.5
- BEIR 2.2.0
- FAISS CPU 1.15.1
- rank-bm25 0.2.2
- pytrec-eval-terrier 0.5.10
- NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1
- pytest 9.1.1, Accelerate 1.15.0

## Validation

`pip check`: no broken requirements.

Offline smoke check passed: all requested package imports, BEIR loader/evaluator
imports, one visible NVIDIA L40S with `CUDA_VISIBLE_DEVICES=0`, CUDA matrix
multiplication matching CPU results, FAISS toy search, BM25 toy ranking, and
perfect-ranking NDCG/recall/reciprocal-rank toy metrics. The latter confirms the
metric library works; full evaluation-harness validation remains a later task.

A deprecated, unused Sentence Transformers losses import generated a warning during
verification and was removed from the smoke script.

## Reproducibility

Portable intent: `environments/retrieval.yml` and
`environments/retrieval-requirements.txt`.
Exact versions: `environments/retrieval-pip-freeze.txt` and
`environments/retrieval-conda-linux-64.explicit.txt`.
See `environments/README.md` for reconstruction and activation commands.

## Remaining scope

No environment blocker was observed. GPU access is confirmed for this environment;
BGE/MedCPT inference, SciFact downloads/mappings, and application evaluation have
not been run. Stop here for the environment-only milestone.
