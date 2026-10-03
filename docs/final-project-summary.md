# Final project summary

## Research question

This project asks how retrieval, evidence selection, and language-model reasoning interact in evidence-grounded biomedical claim verification. The system must retrieve relevant SciFact articles from a 5,183-document corpus and classify a claim as `SUPPORT`, `CONTRADICT`, or `INSUFFICIENT` using only supplied evidence.

The final frozen pipeline is:

```text
claim → zero-shot BGE → exact FAISS top-1 → full abstract
      → Qwen2.5-7B-Instruct → label, citations, explanation
```

BEIR cited-document relevance and original SciFact verification evidence were kept as separate targets throughout the study. This distinction matters because a cited document is not always an annotated verification-evidence document.

## Experimental progression

The work began by validating the BEIR/original-SciFact mapping, tracing claims through document and rationale annotations, and creating a deterministic grouped TRAIN/DEV split. Retrieval metric implementations were checked against trusted libraries and hand-checkable tests.

BM25 established the sparse baseline. Zero-shot `BAAI/bge-base-en-v1.5` substantially improved DEV retrieval and became the primary dense baseline. MedCPT supplied a biomedical dual-encoder comparison.

A controlled training experiment held the BGE checkpoint, loss, explicit-negative count, optimizer, schedule, and steps fixed while changing only negative sampling. Random-negative tuning approximately preserved zero-shot performance. Training with the tested zero-shot-BGE-mined negatives degraded DEV retrieval across three seeds. Cluster-aware analysis found a negative mean effect but appropriately retained uncertainty; the result concerns this mining policy and does not generalize to hard negatives as a class.

Verification development froze Qwen2.5-7B-Instruct, a strict evidence-only JSON prompt, greedy decoding, and an evidence budget. A top-1 full abstract performed better than top-3 or top-5 under this setup. MedCPT cross-encoder reranking improved the BEIR cited-document ranking metric but slightly reduced annotated-evidence top-1 coverage and did not improve downstream verification. Selecting one MedCPT-ranked sentence shortened context substantially but also reduced verification performance. Even a gold annotated rationale sentence did not outperform the annotated full abstract, showing that focused evidence can omit necessary context.

After these choices were frozen, the final zero-shot-BGE, top-1-full-abstract, Qwen pipeline was evaluated once on the official held-out BEIR SciFact TEST queries. No post-TEST tuning or additional architecture evaluation followed.

## Held-out TEST results

| Retrieval metric | Value |
|---|---:|
| NDCG@10 | 0.7404 |
| Recall@10 | 0.8742 |
| Recall@100 | 0.9667 |
| MRR@10 | 0.7034 |

| Verification metric | Final system | Gold-document diagnostic |
|---|---:|---:|
| Accuracy | 0.7000 | 0.8300 |
| Macro F1 | 0.6859 | 0.8003 |

The final system achieved class F1 values of 0.7480 for SUPPORT, 0.6080 for CONTRADICT, and 0.7018 for INSUFFICIENT. The gap to the gold-document diagnostic shows the cost of evidence selection. The diagnostic also remained imperfect, especially on contradiction, which demonstrates a separate verifier-reasoning limitation.

## Serving benchmark

The frozen system was packaged as a local service with FastAPI, exact FAISS retrieval, and a vLLM OpenAI-compatible Qwen endpoint. Corpus embeddings and the index load once at startup. The production endpoint remains fixed at top-1 and validates generated labels and citations without silently repairing malformed output.

On one NVIDIA L40S at concurrency 8:

| Metric | Value |
|---|---:|
| End-to-end throughput | 4.34 requests/s |
| P50 latency | 1.65 s |
| P95 latency | 2.43 s |
| Mean retrieval latency | 21.8 ms |
| Mean FAISS latency | 0.76 ms |
| Peak GPU memory | 34,896 MiB |
| Failure rate | 0% |
| Invalid-output rate | 0% |

Generation accounted for approximately 98% of end-to-end time. Continuous batching increased generator throughput from roughly 47.5 output tokens/s at concurrency 1 to about 337–339 output tokens/s at concurrency 8, with a moderate latency increase.

## Conclusions

- Zero-shot BGE was a strong and robust retrieval baseline for this small benchmark.
- Random-negative tuning did not establish a reliable gain, while the tested model-mined-negative policy consistently hurt DEV retrieval.
- Better performance on the BEIR cited-document target did not guarantee better verification evidence or downstream labels.
- Single-sentence localization frequently removed context needed for biomedical relation and polarity judgments.
- Retrieval remains important, but annotated-document experiments show that verifier reasoning is an independent bottleneck.
- A simple single-GPU architecture can serve the frozen pipeline with useful batching gains and low retrieval overhead.

## Limitations

SciFact is small, and its retrieval and verification annotations represent different objectives. DEV was reused for prompt and evidence-budget development, while only the final selected system received confirmatory TEST evaluation. The negative-mining, reranking, and passage findings apply to the exact methods tested. The gold-document condition is diagnostic rather than a strict upper bound. Serving measurements reflect one L40S, one software stack, and a fixed request mix; they do not establish general production capacity.
