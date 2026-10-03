import json
import pytest
from retrieval.verification import (
    classification_metrics, format_evidence, map_scifact_annotation,
    parse_and_validate_output, validate_dev_only,
)

class WordTokenizer:
    def encode(self, text, add_special_tokens=False): return text.split()
    def decode(self, ids, skip_special_tokens=True): return " ".join(ids)


def test_label_mapping_support_contradict_and_insufficient():
    base={"id":1,"claim":"x","cited_doc_ids":[2]}
    assert map_scifact_annotation({**base,"evidence":{}})["label"]=="INSUFFICIENT"
    support=map_scifact_annotation({**base,"evidence":{"2":[{"sentences":[0],"label":"SUPPORT"}]}})
    assert support["label"]=="SUPPORT" and support["annotated_evidence"]["2"][0]["sentence_ids"]==[0]
    with pytest.raises(ValueError):
        map_scifact_annotation({**base,"evidence":{"2":[{"sentences":[0],"label":"SUPPORT"},{"sentences":[1],"label":"CONTRADICT"}]}})


def test_dev_only_enforcement_is_exact():
    validate_dev_only(["1","2"],["2","1"])
    with pytest.raises(ValueError): validate_dev_only(["1"],["1","2"])
    with pytest.raises(ValueError): validate_dev_only(["1","2","3"],["1","2"])


def test_output_schema_and_citation_validation():
    raw=json.dumps({"label":"SUPPORT","evidence_ids":["7"],"explanation":"The document states it."})
    value,errors=parse_and_validate_output(raw,["7","8"])
    assert not errors and value["label"]=="SUPPORT"
    _,errors=parse_and_validate_output(raw,["8"])
    assert "invented_evidence_id" in errors
    _,errors=parse_and_validate_output('```json '+raw+' ```',["7"])
    assert errors[0].startswith("invalid_json")


def test_insufficient_must_not_cite():
    raw=json.dumps({"label":"INSUFFICIENT","evidence_ids":["7"],"explanation":"Not enough."})
    _,errors=parse_and_validate_output(raw,["7"])
    assert "insufficient_with_citations" in errors


def test_evidence_format_preserves_ids_and_reports_truncation():
    docs=[{"doc_id":"7","title":"A title","abstract":["one two three","four five six"]}]
    text,counts=format_evidence(docs,WordTokenizer(),100)
    assert "[EVIDENCE_ID: 7; SENTENCE: 0]" in text
    assert counts["before"]==counts["after"] and not counts["truncated"]
    text,counts=format_evidence(docs,WordTokenizer(),10)
    assert "[EVIDENCE_ID: 7]" in text and counts["truncated"] and counts["after"]<=10


def test_metric_reconstruction():
    gold=["SUPPORT","CONTRADICT","INSUFFICIENT"]
    result=classification_metrics(gold,gold)
    assert result["accuracy"]==1.0 and result["macro_f1"]==1.0
    assert all(result["per_label"][label]["f1"]==1.0 for label in gold)
