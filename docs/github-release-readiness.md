# GitHub release readiness

## Status

The repository is prepared and staged locally on branch `main`. It has no commits and no remotes. Nothing has been uploaded or pushed.

Recommended repository name: **`biomedical-evidence-verification`**

Suggested description:

> Evidence-grounded biomedical claim verification with BGE, FAISS, Qwen, vLLM, and FastAPI.

Suggested topics:

- `machine-learning`
- `information-retrieval`
- `rag`
- `biomedical-nlp`
- `llm`
- `faiss`
- `vllm`
- `fastapi`

## Repository audit

Public source code, frozen configs, tests, small result summaries, milestone reports, benchmark scripts, portable environment definitions, and exact dependency snapshots are staged. The top-level README now presents the architecture, frozen held-out TEST results, serving measurements, controlled experiments, limitations, reproduction path, and API examples. `docs/final-project-summary.md` provides a shorter research narrative.

The following remain local and ignored rather than being deleted:

- three resume PDFs
- two early proposal/patch documents
- Python bytecode and pytest caches
- datasets and extracted corpora
- model weights and Hugging Face caches
- checkpoints, embeddings, indexes, and raw run outputs
- service request logs and GPU/runtime logs
- Conda environments and temporary files
- archives and snapshots outside the small checked-in summaries

## Secret and privacy scan

A staged-content scan checked for private-key headers, AWS access-key forms, GitHub and Hugging Face token forms, bearer tokens, and password/API-key assignments. It found **zero matches**. No staged email addresses or machine hostnames were found. No `.env`, SSH key, credential file, resume, raw service log, or model artifact is staged.

Eleven files in `docs/` retain `/home/boweiye2/rag/...` artifact references. These are dated historical milestone reports or machine-readable archival summaries. They identify local experiment artifact locations but contain no credentials. Active code, configs, scripts, tests, environment instructions, and README contain no Bowei-specific absolute paths.

## Path portability changes

- `configs/serving-6b-v1.json` now stores paths relative to `RAG_ROOT`.
- `serving/app.py` resolves configured artifacts through `RAG_ROOT`, defaulting to `$HOME/rag`.
- `scripts/validate_serving_6b.py`, `scripts/benchmark_serving_6b.py`, and `scripts/finalize_serving_6b.py` now use `RAG_ROOT` instead of `/home/boweiye2/rag`.
- Five retrieval-derived pip freeze files now record the verified `packaging==26.3` version instead of a nonportable Conda build-worker `file://` URL.
- Historical result documents were not cosmetically rewritten, preserving their original provenance.

These changes affect path discovery and dependency notation only. Model names, revisions, prompts, retrieval settings, benchmark results, and held-out TEST results are unchanged.

## Ignore policy

`.gitignore` covers Python/test caches, local environments, secrets, IDE files, resumes, planning drafts, datasets, model weights, checkpoints, embeddings, indexes, Hugging Face caches, raw runs, logs, binary model formats, large archives, and temporary files. Small result JSON files under `docs/`, frozen configs, and environment definitions remain staged.

## Large-file scan

The staged repository contains no file above 10 MiB, 50 MiB, or 100 MiB. The largest staged file is `configs/splits/scifact_train_dev_v1.json` at 26,675 bytes. Git LFS is unnecessary.

## Validation

- Existing research suite: **34 passed**
- Serving suite: **7 passed**
- Serving environment: `pip check` reports no broken requirements
- Python compile/import syntax check: passed for `retrieval/`, `serving/`, `scripts/`, and `tests/`
- README local-link check: zero missing targets
- Secret scan: zero detected matches
- Active absolute-path scan: zero Bowei-specific paths
- Staged large-file scan: zero files above 10 MiB
- Git remote count: zero
- Staged files after adding this report: **107**

No model inference, training, TEST evaluation, or benchmark was rerun during repository preparation.

## Known limitations

- Full reproduction requires public third-party datasets and model checkpoints that are intentionally not included.
- Several milestone runners refuse to overwrite frozen external run directories and are designed for ordered, milestone-level reproduction rather than a one-command demo.
- Historical reports refer to external raw artifacts that are not part of the public repository.
- Small SciFact examples are included for mapping and audit transparency; SciFact and all third-party model/data assets retain their own licenses.
- The serving benchmark is specific to one L40S and the recorded software/request mix.

## Files requiring user review before the first commit

Review these exact public-facing or licensing-sensitive files:

- `LICENSE` — confirm MIT and the copyright name `Bowei Ye`.
- `README.md` — confirm project framing, claims, links, and reproduction expectations.
- `docs/final-project-summary.md` — confirm the condensed research narrative.
- `docs/github-release-readiness.md` — confirm the public-release audit and recommendations.
- `configs/serving-6b-v1.json` — confirm the relative `RAG_ROOT` artifact layout.
- `docs/scifact-mapping-examples.json` — contains small excerpts from the public SciFact data.
- `docs/training-negative-audit-examples.json` — contains small benchmark-derived examples.

Review whether to retain local provenance paths in these archived files:

- `docs/milestone4c-statistical-failure-analysis.json`
- `docs/milestone5a-verification-dev-report.md`
- `docs/milestone5a-verification-dev-results.json`
- `docs/milestone5b-reranking-dev-report.md`
- `docs/milestone5b-reranking-dev-results.json`
- `docs/milestone5c-passage-localization-dev-report.md`
- `docs/milestone5c-passage-localization-dev-results.json`
- `docs/milestone6a-final-test-report.md`
- `docs/milestone6a-final-test-results.json`
- `docs/milestone6b-serving-report.md`
- `docs/milestone6b-serving-results.json`

They contain no credentials, but they expose the local username and external artifact layout. Keeping them supports provenance; replacing the username with `$RAG_ROOT` would improve cosmetic privacy while changing archival text.

## Final Git state

The index contains only the reviewed repository files under `README.md`, `LICENSE`, `.gitignore`, `configs/`, `retrieval/`, `serving/`, `scripts/`, `tests/`, `docs/`, and `environments/`. The repository has no initial commit and no configured remote. The next action should be a human review of the staged diff, followed by the first local commit if accepted.
