import math
import pytest
import pytrec_eval
from beir.retrieval.evaluation import EvaluateRetrieval
from retrieval.metrics import evaluate

def test_nonperfect_rankings_against_trusted_libraries():
    labels={'multi':{'a':1,'b':1,'c':1}, 'late':{'z':1}, 'missing':{'x':1}, 'edge':{'e':1}}
    runs={'multi':['noise','a','other','b'], 'late':[f'n{i}' for i in range(10)]+['z'], 'missing':[], 'edge':[f'm{i}' for i in range(9)]+['e']}
    means, per=evaluate(labels,runs)
    assert per['multi']['NDCG@10']==pytest.approx((1/math.log2(3)+1/math.log2(5))/(1+1/math.log2(3)+1/math.log2(4)))
    assert per['multi']['Recall@10']==pytest.approx(2/3)
    assert per['late']['MRR@10']==0 and per['late']['Recall@100']==1
    assert per['edge']['MRR@10']==0.1
    assert all(v==0 for v in per['missing'].values())
    scores={q:{d:float(len(ds)-i) for i,d in enumerate(ds)} for q,ds in runs.items()}
    # pytrec_eval omits empty runs: verify those separately against hand values above.
    trusted=pytrec_eval.RelevanceEvaluator(labels,{'ndcg_cut_10','recall_10','recall_100'}).evaluate(scores)
    for q,values in trusted.items():
        for ours,theirs in [('NDCG@10','ndcg_cut_10'),('Recall@10','recall_10'),('Recall@100','recall_100')]:
            assert per[q][ours]==pytest.approx(values[theirs])
    mrr=EvaluateRetrieval.evaluate_custom(labels,scores,[10],metric='mrr')
    assert means['MRR@10']==pytest.approx(mrr['MRR@10'],abs=1e-5)

def test_reject_duplicate_results():
    with pytest.raises(ValueError): evaluate({'q':{'a':1}}, {'q':['a','a']})

def test_recall_100_boundary_and_missing_query():
    labels={'at100':{'a':1},'at101':{'b':1},'absent':{'c':1}}
    runs={'at100':[f'n{i}' for i in range(99)]+['a'], 'at101':[f'n{i}' for i in range(100)]+['b']}
    _,per=evaluate(labels,runs)
    assert per['at100']['Recall@100']==1
    assert per['at101']['Recall@100']==0
    assert all(v==0 for v in per['absent'].values())
