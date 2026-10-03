"""Frozen held-out TEST validation and ordinary bootstrap helpers."""
from __future__ import annotations
from collections import Counter
import numpy as np
from retrieval.verification import LABELS,classification_metrics


def validate_test_partition(test_ids,development_ids):
    test=list(map(str,test_ids)); development=set(map(str,development_ids))
    if not test or len(test)!=len(set(test)): raise ValueError("TEST IDs must be non-empty and unique")
    overlap=sorted(set(test)&development,key=int)
    if overlap: raise ValueError(f"DEV/TRAIN and TEST overlap: {overlap[:5]}")


def annotation_availability(claims):
    if not claims: raise ValueError("No official TEST claims")
    if any("evidence" not in x for x in claims): return False,"evidence_field_missing_or_mixed"
    nonempty=[x for x in claims if x.get("evidence")]
    if not nonempty: return False,"all_evidence_fields_empty_labels_withheld"
    for claim in nonempty:
        for sets in claim["evidence"].values():
            for item in sets:
                if item.get("label") not in {"SUPPORT","CONTRADICT"} or not item.get("sentences"):
                    return False,"incomplete_evidence_annotations"
    return True,"explicit_labeled_evidence_available"


def query_bootstrap(values,replicates,seed):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or replicates<1: raise ValueError("Invalid bootstrap input")
    rng=np.random.default_rng(seed); samples=rng.integers(0,len(values),size=(replicates,len(values)))
    stats=values[samples].mean(axis=1)
    return {"lower":float(np.percentile(stats,2.5)),"upper":float(np.percentile(stats,97.5)),"replicates":replicates,"seed":seed}


def macro_f1_bootstrap(gold,predicted,replicates,seed):
    if len(gold)!=len(predicted) or not gold: raise ValueError("Invalid aligned labels")
    rng=np.random.default_rng(seed); values=[]; n=len(gold)
    for indices in rng.integers(0,n,size=(replicates,n)):
        values.append(classification_metrics([gold[i] for i in indices],[predicted[i] for i in indices])["macro_f1"])
    return {"lower":float(np.percentile(values,2.5)),"upper":float(np.percentile(values,97.5)),"replicates":replicates,"seed":seed}


def confusion_matrix(gold,predicted):
    columns=list(LABELS)+sorted(set(predicted)-set(LABELS))
    return {g:{p:sum(a==g and b==p for a,b in zip(gold,predicted)) for p in columns} for g in LABELS}
