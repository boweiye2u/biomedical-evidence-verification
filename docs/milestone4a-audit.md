# Milestone 4A — MedCPT baseline and training-data audit

Date: 2026-10-02. **No training has been run.** Data and configuration are prepared
for review; they are not evidence that fine-tuning will improve performance.

## DEV baselines

Same 162 DEV claims, same full 5,183-document corpus, unchanged cited-document qrels.

| System | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.6982 | 0.8276 | 0.9144 | 0.6718 |
| BGE zero-shot | 0.7604 | 0.8938 | 0.9815 | 0.7279 |
| MedCPT | 0.7181 | 0.8321 | 0.9691 | 0.6942 |

MedCPT is below BGE here. These DEV results do not establish significance or
held-out generalization. NDCG/recall checked query-by-query against pytrec_eval;
MRR@10 checked against BEIR. Pinned revisions and environment match milestone 3.

MedCPT uses separate CLS query/article encoders, unnormalized inner product,
64-token queries, 512-token title/abstract pairs. Float32, one L40S, four CPU
threads, article batches of 32; five query warmups and 162 sequential batch-1
queries. Timings include query tokenization, GPU encoding, transfer, and exact CPU
FAISS search, with GPU synchronization. They exclude startup and serialization.

Corpus encoding: 20.24 s. Query mean/P50/P95: 5.35/5.34/5.48 ms. Peak PyTorch allocated memory: 1407.0 MiB, excluding non-PyTorch allocations.

## Frozen matched data

- 647 TRAIN queries, **737 positive pairs per arm**, five explicit negatives each.
- 63 queries have multiple positive documents; every positive generates a row.
- Positives are BEIR cited-document relevance, not necessarily verification evidence.
- Random: seeded corpus permutation per query (20261002 + numeric query ID).
- Hard: top 50 from frozen zero-shot BGE, take first five eligible documents.
- Same positive pairs, row order, count, and filtering in both arms.
- All known query positives and their normalized-content duplicates excluded.
- Negative content deduplicated within each query; no duplicates found in corpus
  under lowercase/whitespace-normalized title+abstract fingerprinting.
- Zero known-positive collisions; zero DEV queries used for mining.
- Five random/hard negative slots overlap across the 647 query pools; this is allowed.
- Negative documents may be cited by other queries: 263 random and 297 hard slots
  (unique query pools, not repeated positive-pair rows). This does not itself make
  them relevant to the current query.

Corpus documents are shared and can be candidates regardless of split; no DEV
labels were used for mining or filtering. Candidate lists and scores are retained.

## Sample review and unresolved label uncertainty

Seeded sample of eight training claims; inspected the first two hard negatives per
claim (16 abstracts), plus random-negative titles and available abstract context.
This is an assistant review, not an independently adjudicated human annotation set.
See [example notes](training-negative-audit-examples.json). Full source abstracts
stay outside the repository in `~/rag/runs/milestone4a/audit-examples-full.json`.

Three concrete ambiguity flags:

| Claim | Negative document | Concern |
|---|---|---|
| 316 | 21181273 | PGE2 degradation mechanism may be relevant despite missing qrel |
| 1214 | 22621251 | Monocyte-induced fibrosis may provide contradictory evidence |
| 1119 | 26710772 | Early-pregnancy sympathetic activation is relevant partial evidence |

Other sampled hard negatives often match topic but differ in entity, mechanism,
population, or disease. This small, top-ranked sample cannot estimate a reliable
false-negative rate. **Non-positive qrels do not establish nonrelevance.** We have
not removed flagged examples selectively or changed gold labels. Before 4B, decide
whether to accept this documented benchmark-label noise or introduce a uniformly
specified filtering policy and regenerate/version both arms. This is a research
choice to review, not a technical execution failure.

## Controlled training configuration (not executed)

`configs/training-random-vs-hard-v1.json` records all settings and data hashes.
Proposed simplest clean comparison: standard per-query InfoNCE implemented with
PyTorch cross-entropy, one positive plus five explicit negatives, temperature 0.05.
**In-batch negatives disabled in both arms.** This avoids making another row's
positive/negative an implicit negative for the current query. Cross-query sharing
then creates no in-batch label collision. It does not fix unknown relevance in the
explicit pools. This is deliberately not the standard all-batch MNRL objective;
conclusions must be scoped to the configured explicit-negative experiment.

Same pinned BGE, formatting, normalization, 512-token limits, batch size 16,
accumulation 1, AdamW 2e-5, three epochs, linear schedule with 15 warmup steps,
float32, and matched row shuffles. 47 updates/epoch, 141 total, keeping the final
partial batch. No early stopping. Select the earliest epoch attaining best DEV
NDCG@10. One fixed configuration/seed initially; replicate both arms with the same
three final seeds after development. Each epoch regenerates DEV corpus embeddings
with the current checkpoint; zero-shot embeddings cannot be reused for a trained
model. No trainer or optimizer execution was added in this milestone.

## Artifacts and reproduction

- Data: `~/rag/data/scifact/training_v1/{random,hard}.jsonl` and manifest.
- Raw MedCPT scores, rank-derived metric runs, per-query results and timing:
  `~/rag/runs/milestone4a/`.
- Embeddings/index: `~/rag/embeddings/scifact-medcpt-ea35dd1c515c552d/`.
- Summary and hashes: [milestone4a-results.json](milestone4a-results.json).
- Code: `scripts/prepare_training_4a.py`; tests: `tests/test_training_data.py`.

```bash
source scripts/env.sh
conda activate "$RAG_ROOT/envs/retrieval"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m scripts.prepare_training_4a
python -m pytest -q -p no:cacheprovider tests/
```

A rerun reuses embeddings and checks frozen training file contents before reuse.
It overwrites named timing/result outputs, so preserve the first run before rerunning
for comparative timings. SHA-256 hashes are in the manifest; split and source-file
hashes are checked before processing. No test qrels were loaded.
