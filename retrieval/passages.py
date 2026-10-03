"""Deterministic sentence-passage helpers for Milestone 5C."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np


def sentence_passages(document):
    required={"doc_id","title","abstract"}
    if not required<=set(document): raise ValueError("Malformed document")
    if not isinstance(document["abstract"],list) or not all(isinstance(x,str) and x for x in document["abstract"]):
        raise ValueError("Abstract must be a non-empty sentence list")
    return [{"document_id":str(document["doc_id"]),"title":document["title"],"sentence_index":i,"text":text}
            for i,text in enumerate(document["abstract"])]


def validate_sentence_preservation(document,passages):
    expected=sentence_passages(document)
    if passages!=expected: raise ValueError(f"Sentence loss, reordering, or modification for {document['doc_id']}")


def rationale_pairs(mapping_record):
    return {(str(doc),int(i)) for doc,sets in mapping_record["annotated_evidence"].items()
            for item in sets for i in item["sentence_ids"]}


def select_gold_rationale(mapping_record,corpus):
    choices=[]
    for document_id,evidence_sets in mapping_record["annotated_evidence"].items():
        if document_id not in corpus: raise ValueError(f"Missing rationale document {document_id}")
        for item in evidence_sets:
            ids=tuple(map(int,item["sentence_ids"]))
            if not ids or len(ids)!=len(set(ids)): raise ValueError("Malformed rationale set")
            if any(i<0 or i>=len(corpus[document_id]["abstract"]) for i in ids): raise ValueError("Rationale index out of range")
            choices.append((len(ids),int(document_id),ids,str(document_id)))
    if not choices: return []
    _,_,ids,document_id=min(choices)
    doc=corpus[document_id]
    return [{"document_id":document_id,"title":doc["title"],"sentence_index":i,"text":doc["abstract"][i]} for i in ids]


def rank_passages(passages,scores):
    if len(passages)!=len(scores) or not passages: raise ValueError("Passage/score mismatch")
    if not np.isfinite(np.asarray(scores,dtype=float)).all(): raise ValueError("Non-finite score")
    return [p for _,p in sorted(enumerate(passages),key=lambda x:(-float(scores[x[0]]),x[0]))]


def format_passage_evidence(passages,tokenizer,max_evidence_tokens):
    if not passages: text="[NO EVIDENCE DOCUMENTS PROVIDED]"
    else:
        grouped=OrderedDict()
        for p in passages:
            key=str(p["document_id"])
            if key not in grouped: grouped[key]={"title":p["title"],"items":[]}
            if grouped[key]["title"]!=p["title"]: raise ValueError("Document/title mismatch")
            grouped[key]["items"].append(p)
        blocks=[]
        for doc,g in grouped.items():
            lines=[f"[EVIDENCE_ID: {doc}] Title: {g['title']}"]
            lines.extend(f"[EVIDENCE_ID: {doc}; SENTENCE: {p['sentence_index']}] {p['text']}" for p in g["items"])
            blocks.append("\n".join(lines))
        text="\n\n".join(blocks)
    count=len(tokenizer.encode(text,add_special_tokens=False))
    if count>max_evidence_tokens: raise ValueError("Focused passage evidence unexpectedly exceeds frozen budget")
    return text,{"before":count,"after":count,"truncated":False}
