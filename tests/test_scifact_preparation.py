import importlib.util
from pathlib import Path
import json
import pytest

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).parents[1] / 'scripts/prepare_scifact.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def claim(text, cited, evidence=None):
    return {'claim': text, 'cited_doc_ids': cited, 'evidence': evidence or {}}

def test_related_claims_stay_together():
    claims = {'1':claim('alpha beta gamma', [10]), '2':claim('opposite finding', [10,20]),
              '3':claim('other result', [20]), '4':claim('ALPHA beta gamma!', [30]),
              '5':claim('distinct topic', [40])}
    groups = m.groups_for(claims)
    assert groups == [['1','2','3','4'],['5']]
    a, b = m.split_groups(groups, fraction=0.4)
    assert not set(a)&set(b)
    assert all(set(g)<=set(a) or set(g)<=set(b) for g in groups)

@pytest.mark.skipif(not (m.DATA / "original/claims_train.jsonl").exists(), reason="Run SciFact preparation first")
def test_real_split_and_mapping():
    split=json.loads((m.ROOT/'configs/splits/scifact_train_dev_v1.json').read_text())
    claims=m.rows(m.DATA/'original/claims_train.jsonl','id')
    groups=m.groups_for(claims)
    train,dev=m.split_groups(groups)
    assert train==split['train_ids'] and dev==split['dev_ids']
    assert len(train)==647 and len(dev)==162
    assert set(train)|set(dev)==set(claims)
    train_docs={str(d) for q in train for d in claims[q]['cited_doc_ids']}
    dev_docs={str(d) for q in dev for d in claims[q]['cited_doc_ids']}
    assert not train_docs&dev_docs
    for name,sha in split['input_sha256'].items(): assert m.digest(m.DATA/name)==sha
    examples=json.loads((m.ROOT/'docs/scifact-mapping-examples.json').read_text())
    assert len(examples)==10
    assert all(e['query_id'] in train for e in examples)
    assert any(not e['evidence'] and e['qrel_document_ids'] for e in examples)
