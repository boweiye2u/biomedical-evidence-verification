import json
from copy import deepcopy
from pathlib import Path
import pytest
from posttraining.data.build_m2_mixtures import hamilton_counts,sample_mixture,validate_pool

def row(pool,i,claim=None):
 condition={"A":"A_gold","B":"B_retrieved_non_gold","C":"C_hard_negative","D":"D_random_negative"}[pool];positive=pool=="A"
 return {"claim_id":str(claim or i),"doc_id":str(1000+i),"condition":condition,"source_split":"TRAIN","claim":"c","evidence":{"title":"t","abstract":["s"]},"claim_gold_label":"SUPPORT","pair_annotation":"SUPPORT" if positive else None,"context_label":"SUPPORT" if positive else "INSUFFICIENT","rationale_sentences":[0] if positive else [],"annotation_basis":"frozen_random_unrelated_pair_seed_20261010" if pool=="D" else "x","retrieval_rank":1 if pool in "BC" else None,"retrieval_score":0.5 if pool=="B" else None,"filter_version":f"pool-{pool.lower()}-frozen-v1" if pool in "BC" else None,"filter_status":"approved" if pool=="B" else None}

def specs(pool,n):
 d={"rows":n,"condition":{"A":"A_gold","B":"B_retrieved_non_gold","C":"C_hard_negative","D":"D_random_negative"}[pool]}
 if pool in "BC":d["required_filter_version"]=f"pool-{pool.lower()}-frozen-v1"
 if pool=="B":d["maximum_score_inclusive"]=.66
 if pool=="D":d["required_source_seed"]=20261010
 return d

def test_hamilton_counts_match_frozen_candidates_without_oversampling():
 assert hamilton_counts({"A":.45,"B":.2,"C":.2,"D":.15},{"A":456,"B":86,"C":5716,"D":647})==(430,{"A":194,"B":86,"C":86,"D":64})
 assert hamilton_counts({"A":.6,"B":.15,"C":.15,"D":.1},{"A":456,"B":86,"C":5716,"D":647})==(573,{"A":344,"B":86,"C":86,"D":57})

def test_sampling_is_reproducible_and_has_no_duplicate_pairs():
 pools={p:[row(p,i+10000*j) for i in range(1,n+1)] for j,(p,n) in enumerate({"A":20,"B":10,"C":10,"D":10}.items())}
 spec={"percentages":{"A":.4,"B":.2,"C":.2,"D":.2},"sampling_seed":7,"expected_total":50,"expected_counts":{"A":20,"B":10,"C":10,"D":10}}
 x=sample_mixture(pools,"x",spec);y=sample_mixture(pools,"x",spec)
 assert x==y and len({(r['claim_id'],r['doc_id']) for r in x})==len(x)
 assert all(r['mixture_name']=="x" and r['sampling_seed']==7 for r in x)

def test_pool_validation_enforces_train_split_and_dev_exclusion():
 r=row("A",1);validate_pool("A",[r],specs("A",1),{"1"},{"2"})
 bad=deepcopy(r);bad["source_split"]="DEV"
 with pytest.raises(ValueError,match="leakage"):validate_pool("A",[bad],specs("A",1),{"1"},{"2"})
 with pytest.raises(ValueError,match="leakage"):validate_pool("A",[r],specs("A",1),{"1"},{"1"})

def test_semantics_and_rationale_schema_are_enforced():
 for p in "ABCD":validate_pool(p,[row(p,1)],specs(p,1),{"1"},set())
 bad=row("C",1);bad["rationale_sentences"]=[0]
 with pytest.raises(ValueError,match="semantics"):validate_pool("C",[bad],specs("C",1),{"1"},set())
 bad=row("A",1);bad["pair_annotation"]="CONTRADICT"
 with pytest.raises(ValueError,match="semantics"):validate_pool("A",[bad],specs("A",1),{"1"},set())

def test_b_c_d_provenance_is_frozen():
 bad=row("B",1);bad["retrieval_score"]=.7
 with pytest.raises(ValueError,match="Unapproved"):validate_pool("B",[bad],specs("B",1),{"1"},set())
 bad=row("C",1);bad["filter_version"]="other"
 with pytest.raises(ValueError,match="filter"):validate_pool("C",[bad],specs("C",1),{"1"},set())
 bad=row("D",1);bad["annotation_basis"]="other"
 with pytest.raises(ValueError,match="seed"):validate_pool("D",[bad],specs("D",1),{"1"},set())

def test_repository_config_disables_benchmark_and_mix3():
 root=Path(__file__).resolve().parents[2];cfg=json.loads((root/'configs/posttraining/m2-mixtures-v1.json').read_text())
 assert cfg['scope']=={'source_split':'TRAIN_only','dev_or_fixed_benchmark_access_allowed':False}
 assert set(cfg['candidates'])=={'mix-1','mix-2'} and cfg['pool_decisions']=={'B':'retain','C':'retain','mix_3_active':False}
 assert all(len(x['sha256'])==64 for x in cfg['source_pools'].values())
