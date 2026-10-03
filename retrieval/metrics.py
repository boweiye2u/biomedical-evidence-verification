"""Binary-relevance metrics for SciFact, with explicit rank cutoffs."""
import math

def evaluate(qrels, runs):
    per_query = {}
    for q, labels in qrels.items():
        positives = {d for d, rel in labels.items() if rel > 0}
        if not positives:
            raise ValueError('Each evaluated query must have positive qrels')
        ranking = runs.get(q, [])
        if len(ranking) != len(set(ranking)):
            raise ValueError('Duplicate document in ranking')
        hits = [i+1 for i,d in enumerate(ranking) if d in positives]
        dcg = sum(1/math.log2(i+1) for i in hits if i <= 10)
        ideal = sum(1/math.log2(i+1) for i in range(1,min(10,len(positives))+1))
        per_query[q] = {
            'NDCG@10': dcg/ideal,
            'Recall@10': sum(i<=10 for i in hits)/len(positives),
            'Recall@100': sum(i<=100 for i in hits)/len(positives),
            'MRR@10': 1/hits[0] if hits and hits[0]<=10 else 0.0,
        }
    if not per_query: raise ValueError('Empty evaluation set')
    mean = {k: sum(r[k] for r in per_query.values())/len(per_query) for k in next(iter(per_query.values()))}
    return mean, per_query
