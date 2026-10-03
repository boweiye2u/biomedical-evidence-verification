import json
import pytest
from scripts.prepare_training_4a import select_negatives, ART, ROOT

def test_negative_filter_excludes_all_positives_and_content_duplicates():
    fp={'p':'positive','p2':'positive2','copy':'positive','a':'a','acopy':'a','b':'b'}
    assert select_negatives(['p','copy','p2','a','acopy','b'],{'p','p2'},fp,2)==['a','b']
    with pytest.raises(ValueError): select_negatives(['p','copy'],{'p'},fp,1)

@pytest.mark.skipif(not (ART/'data/scifact/training_v1/random.jsonl').exists(),reason='Run milestone 4A first')
def test_matched_training_rows_and_no_dev_queries():
    folder=ART/'data/scifact/training_v1'
    random=[json.loads(line) for line in (folder/'random.jsonl').read_text().splitlines()]
    hard=[json.loads(line) for line in (folder/'hard.jsonl').read_text().splitlines()]
    split=json.loads((ROOT/'configs/splits/scifact_train_dev_v1.json').read_text())
    assert len(random)==len(hard)>647
    assert {(r['query_id'],r['positive_id']) for r in random}=={(r['query_id'],r['positive_id']) for r in hard}
    assert [(r['query_id'],r['positive_id']) for r in random]==[(r['query_id'],r['positive_id']) for r in hard]
    assert {r['query_id'] for r in random}==set(split['train_ids'])
    assert not {r['query_id'] for r in random}&set(split['dev_ids'])
    for row in random+hard:
        assert len(set(row['negative_ids']))==5
        assert row['positive_id'] not in row['negative_ids']
