# SciFact DEV retrieval baselines — 2026-10-01

Milestone 3 completed: validated metrics, BM25 and zero-shot BGE DEV runs,
and MedCPT sanity checks. No test judgments, fine-tuning, negative mining,
reranking, or generation were used.

## Evaluation population

162 development claims from `scifact_train_dev_v1`, split seed 20261001,
searched against all 5,183 SciFact documents. Source files were checked against
the frozen split's SHA-256 hashes. Labels are unchanged BEIR **cited-document
relevance**; these results do not measure verification-evidence coverage.

## Results

| System | NDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|
| bm25 | 0.6982 | 0.8276 | 0.9144 | 0.6718 |
| bge | 0.7604 | 0.8938 | 0.9815 | 0.7279 |

BGE is ahead on this development split. No statistical significance, held-out
generalization, fine-tuning gain, or state-of-the-art claim is established.

## Metric validation

`retrieval/metrics.py` implements binary-relevance NDCG@10, Recall@10,
Recall@100, and explicitly truncated MRR@10. Tests cover a non-perfect ranking,
multiple positives, an omitted positive, empty/missing results, a first hit at
rank 10 versus 11, and the Recall@100 boundary. NDCG/recall agree with
`pytrec_eval`; MRR@10 agrees with BEIR and hand calculations. Every actual DEV
query's NDCG/recall was also cross-checked, and aggregate MRR checked with BEIR.

Raw result JSONs use strictly descending **rank-derived scores** to preserve the
adapter's exact ordering, including ties. They are not similarity scores and must
not be used for score calibration or score-based fusion. All metrics depend only
on ranking. BM25 ties retain ascending numeric document ID; BGE uses FAISS's
returned tie ordering. Top 100 results are retained for each query.

## Baseline definitions

BM25: `rank-bm25` BM25Okapi, k1=1.5, b=0.75, epsilon=0.25; title + space +
abstract, Unicode word tokenization (`\w+`), lowercasing, no stemming or stopword
removal. No parameter tuning. This Python implementation is a transparent baseline,
not a benchmark of an optimized production lexical search engine.

BGE: canonical `BAAI/bge-base-en-v1.5`, 512-token limit, title + space + abstract
with no document prefix, CLS pooling, L2 normalization, float32 inference, eval mode,
and no gradients. Queries use the model-card instruction
`Represent this sentence for searching relevant passages: `.
Full corpus encoded with batch size 32; online query batch size 1. Exact CPU FAISS
IndexFlatIP over normalized embeddings implements cosine search. No approximate index.
TF32 was disabled. Three training-query outputs had shape (3, 768), unit norms,
finite values, and zero observed difference on repeated encoding. All document
embeddings were checked for shape, finiteness, and unit norms.

Training-only nearest-neighbor inspection: claim 2 retrieved its cited prion paper
first; claim 4 retrieved a metastases-related paper first and its cited paper second;
claim 6 retrieved its cited SIDS paper first. No settings were changed based on these.

MedCPT: separate Query and Article encoders, CLS pooling without normalization,
inner-product scores; query max length 64; article max length 512 with title/abstract
tokenizer pairs. Pair segment IDs were checked. Both produced repeatable finite
(3, 768) embeddings for training claims 2, 4, 6 and their cited articles. In the
3-by-3 score matrix, each paired article scored highest for its query. This is a
sanity check only, not a corpus-level quality result.

Implementation sources:
- [BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [MedCPT official usage and inner-product search](https://github.com/ncbi/MedCPT)

## Runtime observations

One L40S (GPU 0), float32, four PyTorch/FAISS CPU threads. Concurrency 1, five
warm-up queries, then one pass of all 162 DEV queries. GPU operations synchronized
at timing boundaries. This is a descriptive local measurement, not a sustained-load
or repeated-run benchmark.

| System | Mean query ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| bm25 | 7.923 | 7.413 | 13.383 |
| bge | 5.539 | 5.518 | 5.642 |

BM25 index construction: 0.442 s. Query timings include
tokenization, corpus scoring, and sorting. BGE document encoding: 20.190 s;
FAISS index construction: 0.000901 s. Mean query encoding:
5.163 ms; mean FAISS search: 0.376 ms.
BGE query timing includes tokenization, encoding, GPU-to-CPU transfer, and search.
Both exclude startup, model downloads, index construction, result serialization,
and metric computation. CPU BM25 versus GPU encoding + CPU FAISS is not an
architecture-neutral speed comparison.

Peak PyTorch allocated memory during BGE work: 1036.2 MiB
(excludes CUDA context and non-PyTorch allocations). Embedding array:
15,922,176 bytes; serialized flat index stores a further copy.

## Reproducibility and artifacts

Exact encoder revisions:
- `BAAI/bge-base-en-v1.5`: `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- `ncbi/MedCPT-Query-Encoder`: `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc`
- `ncbi/MedCPT-Article-Encoder`: `d05a736da4bb84ee4057b7f7999485be6ed85465`

Environment/package versions and unrounded results:
[scifact-dev-baselines.json](scifact-dev-baselines.json).
Pinned revisions: `configs/model-revisions.json`.

External artifacts:
- Model weights/tokenizers: `~/rag/cache/huggingface/`.
- Embeddings, index, corpus ID order, and metadata:
  `~/rag/embeddings/scifact-bge-a53fd0d3b47586f0/`.
- Rankings, per-query metrics/latencies, training-only neighbor examples:
  `~/rag/runs/scifact_dev_v1/`.
- Execution log: `~/rag/logs/dev-baselines.log`.

Cache key incorporates corpus hash, model revision, document IDs/order, formatting,
query prefix, maximum length, pooling, normalization, dtype, batch size, and relevant
library versions. Reuse requires matching metadata; corpus embeddings were computed
once for this run. Subsequent cached runs report no fresh corpus-encoding time.

```bash
source scripts/env.sh
conda activate "$RAG_ROOT/envs/retrieval"
python -m pytest -q -p no:cacheprovider tests/
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_dev_baselines
```

A rerun overwrites the named baseline run outputs and JSON summary; preserve them
under a new run name before changing experimental settings. The Markdown report
records the first completed run and its timing measurements.

## Blockers and limitations

No execution blockers. Models and adapters are ready for the next approved stage.
MedCPT corpus-level performance is unmeasured. Sequence truncation may omit useful
abstract evidence; the documented 512-token policy was fixed, not tuned on DEV.
The grouped DEV split and cited-document target limit comparisons to published
SciFact test scores. No held-out evaluation or generation is implied by these results.

## Milestone 4A follow-up

The full MedCPT DEV baseline is now available in [the 4A audit](milestone4a-audit.md).
The MedCPT sanity-only wording above describes the original milestone 3 run.
