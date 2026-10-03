import pytest
from retrieval.passages import format_passage_evidence,rank_passages,rationale_pairs,select_gold_rationale,sentence_passages,validate_sentence_preservation

class Tokenizer:
    def encode(self,text,add_special_tokens=False): return text.split()

def test_deterministic_sentence_split_and_preservation():
    doc={"doc_id":7,"title":"T","abstract":["First.","Second?"]}
    expected=[{"document_id":"7","title":"T","sentence_index":0,"text":"First."},{"document_id":"7","title":"T","sentence_index":1,"text":"Second?"}]
    assert sentence_passages(doc)==sentence_passages(doc)==expected
    validate_sentence_preservation(doc,expected)
    with pytest.raises(ValueError): validate_sentence_preservation(doc,list(reversed(expected)))

def test_rationale_mapping_and_deterministic_choice():
    corpus={"9":{"doc_id":9,"title":"B","abstract":["b0","b1"]},"10":{"doc_id":10,"title":"A","abstract":["a0","a1","a2"]}}
    record={"annotated_evidence":{"10":[{"sentence_ids":[1,2]}],"9":[{"sentence_ids":[1]}]}}
    assert rationale_pairs(record)=={("10",1),("10",2),("9",1)}
    assert select_gold_rationale(record,corpus)==[{"document_id":"9","title":"B","sentence_index":1,"text":"b1"}]
    assert select_gold_rationale({"annotated_evidence":{}},corpus)==[]

def test_passage_scoring_direction_alignment_and_format():
    passages=[{"document_id":"1","title":"T","sentence_index":0,"text":"a"},{"document_id":"2","title":"U","sentence_index":1,"text":"b"}]
    assert rank_passages(passages,[0.1,2.0])==[passages[1],passages[0]]
    text,counts=format_passage_evidence([passages[1]],Tokenizer(),100)
    assert text=="[EVIDENCE_ID: 2] Title: U\n[EVIDENCE_ID: 2; SENTENCE: 1] b"
    assert counts["truncated"] is False
