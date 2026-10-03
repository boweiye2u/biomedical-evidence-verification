"""DEV-only SciFact claim-verification utilities for Milestone 5A."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

LABELS = ("SUPPORT", "CONTRADICT", "INSUFFICIENT")


def map_scifact_annotation(claim):
    evidence = claim.get("evidence", {})
    labels = {
        item["label"]
        for evidence_sets in evidence.values()
        for item in evidence_sets
    }
    if not labels:
        label = "INSUFFICIENT"
    elif len(labels) == 1:
        label = next(iter(labels))
    else:
        raise ValueError(f"Conflicting verification labels for claim {claim['id']}: {labels}")
    if label not in LABELS:
        raise ValueError(f"Unknown label {label}")
    documents = {}
    for document_id, evidence_sets in evidence.items():
        documents[str(document_id)] = [
            {
                "sentence_ids": list(item["sentences"]),
                "label": item["label"],
            }
            for item in evidence_sets
        ]
    return {
        "claim_id": str(claim["id"]),
        "claim": claim["claim"],
        "label": label,
        "annotated_evidence": documents,
        "annotated_evidence_document_ids": sorted(documents, key=int),
        "cited_document_ids": [str(value) for value in claim.get("cited_doc_ids", [])],
    }


def validate_exact_ids(expected_ids, actual_ids, name="records"):
    expected = list(map(str, expected_ids))
    actual = list(map(str, actual_ids))
    if len(actual) != len(set(actual)):
        raise ValueError(f"Duplicate IDs in {name}")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual), key=int)
        extra = sorted(set(actual) - set(expected), key=int)
        raise ValueError(f"{name} ID mismatch: missing={missing}, extra={extra}")


def validate_dev_only(requested_ids, frozen_dev_ids):
    validate_exact_ids(frozen_dev_ids, requested_ids, "DEV claims")


def normalize_ranking(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(doc) for doc, _ in sorted(value.items(), key=lambda item: -item[1])]
    raise TypeError(f"Unsupported ranking type: {type(value)}")


def format_evidence(documents, tokenizer, max_evidence_tokens):
    """Format full abstracts with stable document/sentence identifiers.

    Documents are never silently dropped. If the token budget is exhausted, the
    remaining document headers are retained and their text is marked truncated.
    """
    if max_evidence_tokens <= 0:
        raise ValueError("max_evidence_tokens must be positive")
    full_blocks = []
    for document in documents:
        doc_id = str(document["doc_id"])
        lines = [f"[EVIDENCE_ID: {doc_id}] Title: {document['title']}"]
        lines.extend(
            f"[EVIDENCE_ID: {doc_id}; SENTENCE: {index}] {sentence}"
            for index, sentence in enumerate(document["abstract"])
        )
        full_blocks.append("\n".join(lines))
    full_text = "\n\n".join(full_blocks) if full_blocks else "[NO EVIDENCE DOCUMENTS PROVIDED]"
    before = len(tokenizer.encode(full_text, add_special_tokens=False))
    if before <= max_evidence_tokens:
        return full_text, {"before": before, "after": before, "truncated": False}

    output = []
    remaining = max_evidence_tokens
    for document, full_block in zip(documents, full_blocks):
        doc_id = str(document["doc_id"])
        header = f"[EVIDENCE_ID: {doc_id}] Title: {document['title']}"
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        if remaining <= len(header_ids) + 2:
            marker = f"[EVIDENCE_ID: {doc_id}] [TRUNCATED: no text fit within evidence budget]"
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            if remaining >= len(marker_ids):
                output.append(marker)
                remaining -= len(marker_ids)
            continue
        lines = [header]
        remaining -= len(header_ids)
        for index, sentence in enumerate(document["abstract"]):
            line = f"[EVIDENCE_ID: {doc_id}; SENTENCE: {index}] {sentence}"
            ids = tokenizer.encode(line, add_special_tokens=False)
            if len(ids) <= remaining:
                lines.append(line)
                remaining -= len(ids)
                continue
            prefix = f"[EVIDENCE_ID: {doc_id}; SENTENCE: {index}] "
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            marker_ids = tokenizer.encode(" [TRUNCATED]", add_special_tokens=False)
            usable = remaining - len(prefix_ids) - len(marker_ids)
            if usable > 0:
                sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)[:usable]
                lines.append(prefix + tokenizer.decode(sentence_ids, skip_special_tokens=True) + " [TRUNCATED]")
                remaining = 0
            break
        output.append("\n".join(lines))
    formatted = "\n\n".join(output)
    after = len(tokenizer.encode(formatted, add_special_tokens=False))
    if after > max_evidence_tokens:
        ids = tokenizer.encode(formatted, add_special_tokens=False)[:max_evidence_tokens]
        formatted = tokenizer.decode(ids, skip_special_tokens=True)
        after = len(tokenizer.encode(formatted, add_special_tokens=False))
    return formatted, {"before": before, "after": after, "truncated": True}


PROMPTS = {
    "baseline-v1": (
        "Use only the evidence documents below. Classify the claim as SUPPORT, "
        "CONTRADICT, or INSUFFICIENT. Cite only supplied DOC identifiers. If the "
        "documents do not establish or refute the claim, choose INSUFFICIENT. "
        "In evidence_ids, copy the exact raw identifier shown after EVIDENCE_ID, such as "
        "12345; never add a DOC_ prefix. Return exactly one JSON object with keys "
        "label, evidence_ids, explanation."
    ),
    "grounded-v2": (
        "You are verifying one biomedical claim using only the supplied evidence. "
        "First determine whether a supplied document directly establishes the same "
        "entities, relation, population, and polarity as the claim. Choose SUPPORT "
        "only when the evidence directly entails the claim; choose CONTRADICT only "
        "when it directly establishes the opposite; otherwise choose INSUFFICIENT. "
        "Do not use outside knowledge. evidence_ids must contain only supplied DOC "
        "identifiers that justify the decision and must be empty for INSUFFICIENT. "
        "Copy each raw identifier exactly as printed after EVIDENCE_ID (for example, use "
        "\"12345\", never \"DOC_12345\"). Return exactly one valid JSON object and no markdown: "
        '{"label":"SUPPORT|CONTRADICT|INSUFFICIENT","evidence_ids":["DOC_ID"],"explanation":"concise evidence-based explanation"}'
    ),
}


def prompt_hash(name):
    return hashlib.sha256(PROMPTS[name].encode()).hexdigest()


def build_messages(claim, evidence_text, prompt_name):
    if prompt_name not in PROMPTS:
        raise KeyError(prompt_name)
    return [
        {"role": "system", "content": PROMPTS[prompt_name]},
        {"role": "user", "content": f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence_text}"},
    ]


def parse_and_validate_output(raw_output, allowed_document_ids):
    try:
        value = json.loads(raw_output.strip())
    except (json.JSONDecodeError, TypeError) as error:
        return None, [f"invalid_json:{type(error).__name__}"]
    errors = []
    if not isinstance(value, dict):
        return None, ["output_not_object"]
    if set(value) != {"label", "evidence_ids", "explanation"}:
        errors.append("schema_keys")
    if value.get("label") not in LABELS:
        errors.append("invalid_label")
    evidence_ids = value.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
        errors.append("invalid_evidence_ids_type")
    else:
        allowed = set(map(str, allowed_document_ids))
        if len(evidence_ids) != len(set(evidence_ids)):
            errors.append("duplicate_evidence_ids")
        if not set(evidence_ids) <= allowed:
            errors.append("invented_evidence_id")
        if value.get("label") == "INSUFFICIENT" and evidence_ids:
            errors.append("insufficient_with_citations")
    if not isinstance(value.get("explanation"), str) or not value.get("explanation", "").strip():
        errors.append("invalid_explanation")
    return (value if not errors else None), errors


def classification_metrics(gold, predicted):
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Gold and predicted labels must be non-empty and aligned")
    per_label = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        support = sum(g == label for g in gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "accuracy": tp / support if support else None, "support": support}
    return {
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold),
        "macro_f1": sum(per_label[label]["f1"] for label in LABELS) / len(LABELS),
        "per_label": per_label,
        "count": len(gold),
        "prediction_counts": dict(Counter(predicted)),
    }
