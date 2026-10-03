# SciFact data mapping and split — 2026-10-01

Milestone complete. No models downloaded, no retrieval runs, no training.

## Sources and local files

No SciFact assets were present under the project data/cache paths before preparation.
Downloaded only the official BEIR SciFact ZIP and original SciFact data tarball:

- https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
- https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz

Stored under `~/rag/data/scifact`. Archives necessarily include held-out assets;
only the following selected members were extracted:

- `beir/corpus.jsonl`
- `beir/queries.jsonl` (upstream combined query file)
- `beir/qrels/train.tsv`
- `original/corpus.jsonl`
- `original/claims_train.jsonl`

Archives and extracted files occupy approximately **22.75 MB (21.69 MiB)**,
plus a small manifest. Exact per-file byte sizes, source URLs, and SHA-256 hashes
are in [scifact-provenance.json](scifact-provenance.json). Archive URLs include a
mutable `latest` release, so hashes define the inspected snapshot.

## Training mapping findings

All **809 BEIR training query IDs and texts** match original training claims.
All **5,183 corpus IDs and titles** match. Abstracts match after whitespace
normalization; 1,055 documents differ in raw whitespace. Original sentence arrays
must remain authoritative for zero-based rationale sentence indices.

All **919 training qrel pairs equal source-cited documents** (`cited_doc_ids`).
They are NOT uniformly annotated verification evidence:

- 329 claims have cited documents beyond their annotated evidence.
- 304 claims have no annotated evidence at all, despite positive retrieval qrels.
- The remaining 25 such claims have evidence for only some cited documents.
- All annotated evidence documents belong to the corresponding training qrels.
- All rationale indices are in bounds, and labels are SUPPORT or CONTRADICT.
- 616 SUPPORT and 341 CONTRADICT rationale records exist (not claim counts).

Preserve official qrels for the planned BEIR retrieval task. Use only `evidence`
for verification labels, rationale coverage, and evidence-specific metrics.
Do not silently remove the 304 claims or redefine benchmark positives. Retrieval
fine-tuning on all 809 queries therefore learns cited-document relevance, not
exclusively support/contradiction evidence retrieval. This qualifies the project's
scientific wording and is a meaningful reason H4 may not hold.

## Ten manually inspected training-only traces

Full claims, document titles, labels, and rationale text are recorded in
[scifact-mapping-examples.json](scifact-mapping-examples.json).

| Query | BEIR cited/relevant documents | Annotated evidence: document, label, sentence IDs |
|---|---|---|
| 4 | 22942787 | None annotated; do not invent a verification label |
| 6 | 2613775 | None annotated; do not invent a verification label |
| 47 | 26996935, 3512154 | 26996935 SUPPORT [6] |
| 60 | 13899137, 13901073 | 13899137 CONTRADICT [5]; 13899137 CONTRADICT [9]; 13899137 CONTRADICT [11] |
| 2 | 13734012 | 13734012 CONTRADICT [4] |
| 32 | 12428497 | 12428497 CONTRADICT [8] |
| 40 | 13497630 | 13497630 CONTRADICT [7]; 13497630 CONTRADICT [8]; 13497630 CONTRADICT [9]; 13497630 CONTRADICT [10]; 13497630 CONTRADICT [11] |
| 9 | 44265107 | 44265107 SUPPORT [15] |
| 12 | 33409100 | 33409100 SUPPORT [8]; 33409100 SUPPORT [12] |
| 22 | 6490571 | 6490571 SUPPORT [7] |

Examples 47 and 60 demonstrate cited documents without evidence alongside actual
evidence documents. Examples 4 and 6 have no annotated evidence. Examples 2, 32,
and 40 map contradiction rationales; 9, 12, and 22 map support rationales.
Inspection confirms indexing and field alignment, not annotation infallibility.
For example, claim 12 includes a baseline-measurement sentence as a SUPPORT
rationale and broader outcome wording; retain the source label and flag it for
future annotation-quality analysis rather than relabeling it here.

## Frozen split

Saved in `configs/splits/scifact_train_dev_v1.json`:

- Seed: **20261001**; target development fraction: **0.20**.
- **647 train / 162 dev**, covering all 809 BEIR training queries once.
- **490 connected groups**, largest group 8 claims.
- Group edges: shared source-cited or evidence document; exact normalized claim;
  or token-set Jaccard similarity >= 0.8. Connected components are indivisible.
- Sort groups by lowest numeric claim ID, shuffle with the fixed seed, and assign
  a group to dev only when it moves the dev count closer to the target.
- No source-cited documents overlap between train and dev claim groups.
- Train: 407 claims with evidence / 240 without; dev: 98 with / 64 without.
- This is a conservative document-group split, not a random-query baseline split.
  Lexical grouping cannot guarantee absence of every semantic paraphrase.

DEV retrieval must still search the full 5,183-document corpus. Grouping controls
supervised claim relationships, not corpus availability. Never filter the search
corpus to dev cited documents.

## Held-out policy

Original dev/test files and BEIR test qrels were not extracted or read. The BEIR
combined query JSONL is parsed only to identify training IDs; nontraining query
text is not used, displayed, or inspected. Split construction and examples use
training claims only. No held-out claim-label comparison was performed; this
milestone establishes complete coverage of the known training population, not an
independent audit of upstream test-set construction.

## Reproduction and validation

```bash
source scripts/env.sh
conda activate "$RAG_ROOT/envs/retrieval"
python scripts/prepare_scifact.py
python -m pytest -q -p no:cacheprovider tests/test_scifact_preparation.py
```

Cached archives are reused. Existing extracted files and frozen split contents
are checked before reuse; the script refuses to overwrite a changed split.
A full rerun reproduced the split. Two focused tests passed: grouping transitivity
and real-data split coverage, provenance hashes, document separation, and examples.
Mapping validation additionally checks every training query, qrel, and rationale.

## Blockers and decisions

No technical blocker remains for DEV retrieval. Before downstream verification,
define missing annotation versus context insufficiency and retain separate
retrieval-relevance and evidence-coverage metrics. D3 gold evidence is unavailable
for claims with no annotation; they require a separate evaluation condition or
explicitly reported exclusion, never a fabricated rationale.

The next milestone can validate BGE/MedCPT adapters and run DEV baselines. It has
not been started here.
