import pytest
from retrieval.reranking import assert_frozen_generator,build_pairs,cluster_macro_f1_bootstrap,reranked_order,validate_candidate_set

def test_candidate_set_preservation_and_score_direction():
    assert reranked_order(["a","b","c"],[0.1,2.0,1.0])==["b","c","a"]
    assert reranked_order(["a","b","c"],[1.0,1.0,0.0])==["a","b","c"]
    with pytest.raises(ValueError): validate_candidate_set(["a","b"],["a","c"])

def test_deterministic_reranking():
    first=reranked_order(["3","1","2"],[0.2,0.5,0.5])
    assert first==reranked_order(["3","1","2"],[0.2,0.5,0.5])==["1","2","3"]

def test_claim_document_alignment_and_format():
    docs=[{"doc_id":"1","title":"Title","abstract":["First.","Second."]}]
    assert build_pairs("claim",docs)==[["claim","Title First. Second."]]
    with pytest.raises(ValueError): build_pairs("claim",[{"doc_id":"1","title":"x"}])

def test_exact_frozen_generator_validation():
    config={"model":{"name":"q","revision":"r","precision":"bfloat16"},"final_selection":{"prompt_name":"p","prompt_sha256":"h","top_k":1},"generation":{"do_sample":False,"max_new_tokens":192,"seed":7}}
    frozen={"model_name":"q","model_revision":"r","precision":"bfloat16","prompt_name":"p","prompt_sha256":"h","top_k":1,"do_sample":False,"max_new_tokens":192,"seed":7}
    assert_frozen_generator(config,frozen)
    frozen["top_k"]=3
    with pytest.raises(ValueError): assert_frozen_generator(config,frozen)


def test_cluster_macro_f1_bootstrap_reproducible():
    gold=["A","B","A","B"]
    pred_a=["A","A","A","B"]
    pred_b=["A","B","B","B"]
    args=(gold,pred_a,pred_b,["0","1","2","3"],[["0","1"],["2","3"]],["A","B"])
    first=cluster_macro_f1_bootstrap(*args,replicates=100,seed=17)
    second=cluster_macro_f1_bootstrap(*args,replicates=100,seed=17)
    assert first==second
    assert first["replicates"]==100 and first["seed"]==17

    with pytest.raises(ValueError):
        cluster_macro_f1_bootstrap(gold,pred_a,pred_b,["0","1","2","3"],[["0","1"],["1","2"]],["A","B"],replicates=2)
