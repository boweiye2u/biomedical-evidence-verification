import pytest
from retrieval.final_test import annotation_availability,confusion_matrix,macro_f1_bootstrap,query_bootstrap,validate_test_partition

def test_test_only_partition_loading():
    validate_test_partition(["3","4"],["1","2"])
    with pytest.raises(ValueError): validate_test_partition(["2","3"],["1","2"])
    with pytest.raises(ValueError): validate_test_partition(["3","3"],["1"])

def test_withheld_annotations_are_not_insufficient_labels():
    assert annotation_availability([{"id":1,"claim":"x"}])==(False,"evidence_field_missing_or_mixed")
    assert annotation_availability([{"id":1,"claim":"x","evidence":{}}])==(False,"all_evidence_fields_empty_labels_withheld")
    claims=[{"id":1,"claim":"x","evidence":{}},{"id":2,"claim":"y","evidence":{"8":[{"sentences":[0],"label":"SUPPORT"}]}}]
    assert annotation_availability(claims)==(True,"explicit_labeled_evidence_available")

def test_bootstrap_and_confusion_are_reproducible():
    assert query_bootstrap([0,1,1],100,7)==query_bootstrap([0,1,1],100,7)
    assert macro_f1_bootstrap(["SUPPORT","CONTRADICT","INSUFFICIENT"],["SUPPORT","SUPPORT","INSUFFICIENT"],100,8)==macro_f1_bootstrap(["SUPPORT","CONTRADICT","INSUFFICIENT"],["SUPPORT","SUPPORT","INSUFFICIENT"],100,8)
    matrix=confusion_matrix(["SUPPORT","CONTRADICT"],["SUPPORT","SUPPORT"])
    assert matrix["SUPPORT"]["SUPPORT"]==1 and matrix["CONTRADICT"]["SUPPORT"]==1

def test_frozen_config_has_no_test_selection_candidates():
    import json
    from pathlib import Path
    config=json.loads((Path(__file__).parents[1]/"configs/final-test-6a-v1.json").read_text())
    assert config["status"] in {"frozen_before_first_test_record_access","frozen_modeling_protocol_with_documented_post_access_path_correction"}
    assert config["selection_candidates"]==[]
    assert config["retriever"]["revision"]=="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    assert config["verifier"]["prompt_sha256"]=="21c9360a72834e37fcc37936190e290638a924e8985c50bf6dcff99472ba50b4"
