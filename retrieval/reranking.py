"""Deterministic reranking helpers for Milestone 5B."""
from __future__ import annotations
import numpy as np


def validate_candidate_set(original_ids, reranked_ids):
    if len(original_ids)!=len(set(original_ids)) or len(reranked_ids)!=len(set(reranked_ids)):
        raise ValueError("Candidate IDs must be unique")
    if set(original_ids)!=set(reranked_ids):
        raise ValueError("Reranking changed the candidate set")


def reranked_order(candidate_ids, scores):
    if len(candidate_ids)!=len(scores): raise ValueError("Candidate/score length mismatch")
    if not np.isfinite(np.asarray(scores,dtype=float)).all(): raise ValueError("Non-finite reranker score")
    indexed=list(enumerate(zip(map(str,candidate_ids),map(float,scores))))
    ordered=[doc for _,(doc,_) in sorted(indexed,key=lambda x:(-x[1][1],x[0]))]
    validate_candidate_set(list(map(str,candidate_ids)),ordered)
    return ordered


def build_pairs(claim,documents):
    if not isinstance(claim,str) or not claim.strip(): raise ValueError("Claim must be non-empty")
    pairs=[]
    for document in documents:
        if not {"doc_id","title","abstract"}<=set(document): raise ValueError("Malformed document")
        abstract=document["abstract"]
        if not isinstance(abstract,list) or not all(isinstance(x,str) for x in abstract): raise ValueError("Abstract must be sentence list")
        article=(document["title"]+" "+" ".join(abstract)).strip()
        pairs.append([claim,article])
    return pairs


def assert_frozen_generator(config_5a,frozen):
    checks={
      "model_name":config_5a["model"]["name"],"model_revision":config_5a["model"]["revision"],
      "precision":config_5a["model"]["precision"],"prompt_name":config_5a["final_selection"]["prompt_name"],
      "prompt_sha256":config_5a["final_selection"]["prompt_sha256"],"top_k":config_5a["final_selection"]["top_k"],
      "do_sample":config_5a["generation"]["do_sample"],"max_new_tokens":config_5a["generation"]["max_new_tokens"],
      "seed":config_5a["generation"]["seed"],
    }
    if checks!=frozen: raise ValueError(f"Frozen generator mismatch: {checks} != {frozen}")


def macro_f1(labels,predictions,label_names):
    values=[]
    for label in label_names:
        tp=sum(g==label and p==label for g,p in zip(labels,predictions))
        fp=sum(g!=label and p==label for g,p in zip(labels,predictions))
        fn=sum(g==label and p!=label for g,p in zip(labels,predictions))
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
        values.append(2*precision*recall/(precision+recall) if precision+recall else 0.0)
    return sum(values)/len(values)


def cluster_macro_f1_bootstrap(labels,pred_a,pred_b,query_ids,clusters,label_names,replicates=20000,seed=20261012):
    index={q:i for i,q in enumerate(query_ids)}
    flat=[q for c in clusters for q in c]
    if len(flat)!=len(set(flat)) or set(flat)!=set(query_ids): raise ValueError("Invalid clusters")
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(replicates):
        selected=rng.integers(0,len(clusters),size=len(clusters)); positions=[]
        for j in selected: positions.extend(index[q] for q in clusters[j])
        gold=[labels[i] for i in positions]; a=[pred_a[i] for i in positions]; b=[pred_b[i] for i in positions]
        diffs.append(macro_f1(gold,b,label_names)-macro_f1(gold,a,label_names))
    return {"lower":float(np.percentile(diffs,2.5)),"upper":float(np.percentile(diffs,97.5)),"replicates":replicates,"seed":seed}
